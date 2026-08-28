import ast
import pathlib
import unittest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "wait"

# Reading a clock, not merely naming one: probe exposes monotonic_s and now_ns
# for the modules that need a deadline, and they are the sanctioned path.
CLOCKS = frozenset("""
    perf_counter perf_counter_ns monotonic monotonic_ns time time_ns
    process_time process_time_ns thread_time times now utcnow
""".split())
CLOCK_MODULES = frozenset(("time", "os", "datetime", "resource"))

# An explicit allowlist rather than sys.stdlib_module_names, which is 3.10+ while
# the target interpreter is 3.9.18 -- the check has to run where the code runs.
ALLOWED = frozenset("""
    argparse collections concurrent dataclasses enum errno functools hashlib
    importlib itertools json math multiprocessing os pathlib re shutil signal
    socket statistics struct subprocess sys time typing wait
""".split())


def sources():
    return sorted(PACKAGE.rglob("*.py"))


class Hygiene(unittest.TestCase):

    def test_wait_package_imports_stdlib_only(self):
        # /arf forbids pip installs, so a third-party import means the harness
        # cannot run at all -- a failure unrelated to any experiment.
        for path in sources():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    roots = [(node.module or "").split(".")[0]]
                else:
                    continue
                for root in roots:
                    self.assertIn(root, ALLOWED, "%s imports %s" % (path.name, root))

    def test_only_probe_calls_a_clock(self):
        # More than one module timing things means two clocks disagreeing about
        # what a measurement includes.  probe.py owns the clock.
        for path in sources():
            if path.name == "probe.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if (isinstance(func, ast.Attribute) and func.attr in CLOCKS
                        and isinstance(func.value, ast.Name)
                        and func.value.id in CLOCK_MODULES):
                    self.fail("%s calls %s.%s" % (path.name, func.value.id, func.attr))

    def test_no_experiment_writes_a_payload_itself(self):
        # A file whose layout was set by setstripe must not then be truncated:
        # it stays on the MDT, reports DoM, issues no OST RPC, and quietly costs
        # a full round trip instead of a memcpy.  Creation goes through
        # probe.write_files, probe.write_staged or lustre.create_with_layout, so
        # the rule has one place to live.
        for path in sources():
            if "experiments" not in str(path):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "open"):
                    continue
                mode = node.args[1] if len(node.args) > 1 else None
                # Binary write only: a payload.  Text mode is a readings file,
                # which carries no layout and no measurement.
                text = str(mode.value) if isinstance(mode, ast.Constant) else ""
                if "b" in text and ("w" in text or "a" in text):
                    self.fail("%s writes a payload directly" % path.name)


if __name__ == "__main__":
    unittest.main()
