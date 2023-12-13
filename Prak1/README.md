# hpc_praktikum
## Wichtige Befehle:
### Login
$ ssh <zih-login>@login1.barnard.hpc.tu-dresden.de
### GCC Versionen
$ module spider GCC
### GCC laden
$ module load GCC/12.2.0
### Compute-Knoten allokieren
$ salloc -n 1 -c 1 -t 1:00:00  
oder  
$ salloc -n 1 -c 1 -t 1:00:00 --reservation=<reservation_name> -A <group_name> --exclusive
### Programm auf Konten ausführen
$ srun ./my_command
