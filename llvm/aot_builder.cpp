// All comments are in English as requested.
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CodeGen.h" 
#include "llvm/TargetParser/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/CodeGen/CommandFlags.h" // ok to include; not strictly required
#include <cstdio>
#include <memory>

using namespace llvm;

static std::unique_ptr<Module> buildModule(LLVMContext& Ctx, const std::string& TripleStr) {
    auto M = std::make_unique<Module>("aot_mod", Ctx);
    M->setTargetTriple(llvm::Triple(TripleStr));

    // Build: extern "C" double foo(double a, double b) { return a + b; }
    Type* DoubleTy = Type::getDoubleTy(Ctx);
    FunctionType* FT = FunctionType::get(DoubleTy, { DoubleTy, DoubleTy }, false);
    // Use ExternalLinkage + unmangled C name so runner can 'extern "C"' it
    Function* F = Function::Create(FT, Function::ExternalLinkage, "foo", M.get());
    F->getArg(0)->setName("a");
    F->getArg(1)->setName("b");

    BasicBlock* BB = BasicBlock::Create(Ctx, "entry", F);
    IRBuilder<> B(BB);
    Value* Sum = B.CreateFAdd(F->getArg(0), F->getArg(1), "sum");
    B.CreateRet(Sum);

    if (verifyFunction(*F, &errs())) {
        errs() << "Function verification failed\n";
        return nullptr;
    }
    return M;
}

static bool emitObj(Module& M, const std::string& OutObjPath) {
    std::string Error;
    auto TT = Triple(M.getTargetTriple());
    const Target* T = TargetRegistry::lookupTarget(TT.getTriple(), Error);
    if (!T) {
        errs() << "lookupTarget failed: " << Error << "\n";
        return false;
    }

    // Configure a generic native TargetMachine
    std::string CPU = sys::getHostCPUName().str(); // or "generic"
    std::string Features = "";                      // host features if desired
    TargetOptions Options;
    auto RM = std::optional<Reloc::Model>(); // default
    auto TM = std::unique_ptr<TargetMachine>(
        T->createTargetMachine(TT.getTriple(), CPU, Features, Options, RM));

    // DataLayout must match the TM
    M.setDataLayout(TM->createDataLayout());

    // Open output .obj file
    std::error_code EC;
    raw_fd_ostream Dest(OutObjPath, EC, sys::fs::OF_None);
    if (EC) {
        errs() << "Could not open file: " << OutObjPath << " : " << EC.message() << "\n";
        return false;
    }

    // Use the legacy PM to emit an object file
    legacy::PassManager PM;
    if (TM->addPassesToEmitFile(PM, Dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        errs() << "TargetMachine can't emit a file of this type\n";
        return false;
    }
    PM.run(M);
    Dest.flush();
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: aot_builder <output.obj>\n");
        return 1;
    }
    std::string OutObj = argv[1];

    // Initialize native targets (AOT requires codegen backends)
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    LLVMContext Ctx;
    std::string TripleStr = sys::getDefaultTargetTriple();
    auto Mod = buildModule(Ctx, TripleStr);
    if (!Mod) return 2;

    if (!emitObj(*Mod, OutObj)) return 3;

    outs() << "Wrote object: " << OutObj << "\n";
    return 0;
}
