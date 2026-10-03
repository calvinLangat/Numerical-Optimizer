To compile unconstrained optimizer
gcc -o test.exe .\main.c .\optimizer.c ..\utils\utils.c

To compile interior point constrained optimizer:
gcc -o testConst -O2 .\main_const.c .\optimizer_const.c ..\utils\utils.c