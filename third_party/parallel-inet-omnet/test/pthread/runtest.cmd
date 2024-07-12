@echo off
rem
rem usage: runtest [<testfile>...]
rem without args, runs all *.test files in the current directory
rem

set TESTFILES=%*
if "x%TESTFILES%" == "x" set TESTFILES=*.test

path %~dp0\..\bin;%PATH%
mkdir work 2>nul

call opp_test %OPT% -g -v %TESTFILES% || goto end

cd work || goto end
call opp_nmakemake -f -e cc --deep --no-deep-includes -I../../../src/common -I../../../src/envir -I../../../include/platdep -I $(PTHREADS_DIR)/include -lpthreadVC2static -L$(PTHREADS_DIR)/lib || goto end
nmake -f makefile.vc SHARED_LIBS=no || cd .. && goto end
cd .. || goto end

call opp_test %OPT% -r -v %TESTFILES% || goto end

echo.
echo Results can be found in work/

:end