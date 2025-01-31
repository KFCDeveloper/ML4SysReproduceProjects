#!/bin/bash

for file in *.png *.pdf; do
  if [[ -f $file ]]; then
    rm "$file"
    echo "Deleted: $file"
  fi
done