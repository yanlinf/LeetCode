#!/bin/bash

for file in *.py
do
    if [ $file != "complete.py" ]
    then
        echo -e "# $file\n" >> complete.py
        cat $file >> complete.py
        echo -e "\n\n\n" >> complete.py
    fi
done

