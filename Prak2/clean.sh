#!/bin/bash

read -p "Wenn du wirklich alle Textdatein löschen willst schreib "1" " loeschen
if [ $loeschen -eq 1 ]
then
  echo "Lösche alle Textdatein mit matmul_[...].txt"
  # entfernt alle Dateiten die mit matmul_ beginnen und mit .txt enden
  ls -a | grep -P "matmul_.*.txt" | xargs -d"\n" rm
else
  echo "Abbruch"
  exit
fi
