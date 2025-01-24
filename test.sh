#!/bin/zsh

echo "Removing prior test-run data:"
rm -rf data/*
echo "Processing, inspecting, comparing, and .. if all goes well, querying:"
sh process.sh && sh inspect.sh && sh compare.sh && sh query.sh
echo "End of test"
