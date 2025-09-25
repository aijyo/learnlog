# -*- coding: utf-8 -*-
# All comments are in English.

import tvm
from tvm.script import ir as I
from tvm.script import tir as T

# -----------------------------------------------------------------------------
# 1) TIR IRModule (use Buffer parameters in signature to avoid match_buffer)
# -----------------------------------------------------------------------------
@I.ir_module
class MyMod:
    @T.prim_func
    def add_one(a: T.Buffer[(16,), "int32"], b: T.Buffer[(16,), "int32"]):
        """Compute b[i] = a[i] + 1 with intentional no-ops."""
        for i in T.serial(16):
            with T.block("B"):
                vi = T.axis.spatial(16, i)
                b[vi] = a[vi] + T.int32(1) + T.int32(0) - T.int32(0)

mod = MyMod

# -----------------------------------------------------------------------------
# 2) Custom PassInstrument via decorator
# -----------------------------------------------------------------------------
@tvm.instrument.pass_instrument
class DemoInstrument:
    def __init__(self, skip_keyword: str | None = None):
        self.skip_keyword = skip_keyword
        self.run_count = 0
        self.before = []
        self.after = []

    def enter_pass_ctx(self):
        print("[DemoInstrument] Enter pass context")

    def exit_pass_ctx(self):
        print(f"[DemoInstrument] Exit. passes seen: {self.run_count}")
        print(f"[DemoInstrument] before: {self.before}")
        print(f"[DemoInstrument] after : {self.after}")

    def should_run(self, mod, pass_info: tvm.transform.PassInfo) -> bool:
        if self.skip_keyword and self.skip_keyword in pass_info.name:
            print(f"[DemoInstrument] Skip pass: {pass_info.name}")
            return False
        return True

    def run_before_pass(self, mod, pass_info: tvm.transform.PassInfo):
        self.run_count += 1
        self.before.append(pass_info.name)
        print(f"[DemoInstrument] BEFORE: {pass_info.name}")

    def run_after_pass(self, mod, pass_info: tvm.transform.PassInfo):
        self.after.append(pass_info.name)
        print(f"[DemoInstrument] AFTER : {pass_info.name}")

# -----------------------------------------------------------------------------
# 3) Pass pipeline (safe on this tiny TIR)
# -----------------------------------------------------------------------------
pipeline = tvm.transform.Sequential(
    [
        tvm.tir.transform.Simplify(),
        tvm.tir.transform.RemoveNoOp(),
    ],
    name="SimpleTIRPipeline",
)

# -----------------------------------------------------------------------------
# 4) Built-in instruments
# -----------------------------------------------------------------------------
printing_inst = tvm.instrument.PassPrintingInstrument(
    print_before_pass_names=["tir.Simplify"],
    print_after_pass_names=["tir.RemoveNoOp"],
)
timing_inst = tvm.instrument.PassTimingInstrument()
demo_inst = DemoInstrument(skip_keyword="RemoveNoOp")  # try skipping RemoveNoOp once

print("\n=== RUN #1: skip 'RemoveNoOp' ===")
with tvm.transform.PassContext(opt_level=3, instruments=[demo_inst, printing_inst, timing_inst]):
    mod_after = pipeline(mod)

print("\n[PassTimingInstrument] Profile #1")
print(timing_inst.render())

print("\n=== RUN #2: no skipping + very verbose ===")
demo_inst2 = DemoInstrument(skip_keyword=None)
with tvm.transform.PassContext(
    opt_level=3,
    instruments=[
        demo_inst2,
        tvm.instrument.PrintBeforeAll(),  # very verbose
        tvm.instrument.PrintAfterAll(),   # very verbose
        tvm.instrument.PassTimingInstrument(),
    ],
):
    mod_after2 = pipeline(mod)

print("\n[INFO] Done.")
