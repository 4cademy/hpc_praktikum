#!/bin/bash

# entfernt alle Dateiten die mit matmul_ beginnen und mit .txt enden
ls -a | grep -P "matmul_.*.txt" | xargs -d"\n" rm