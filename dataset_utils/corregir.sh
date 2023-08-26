#!/bin/bash

for file in I*; do
  lowercase_name=$(echo "$file" | sed 's/\á/a/')
    if [[ "$file" != "$lowercase_name" ]]; then
	        mv -- "$file" "$lowercase_name"
		  fi
	  done
