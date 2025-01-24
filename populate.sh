#!/bin/bash
#
# populate.sh - puts some files in archive/testCollection/ for the purposes
# of having some input data for use in the process.sh stage.

# In the final form, a directory of 80,000+ PDF files will be processed.
#
# This is impractical for development/hacking.
#
# Therefore, this script can be used to copy 50 random files from the main 
# PDF Archive, for the purposes of testing/development/etc.
#


# Directory containing the files
SOURCE_DIR="$HOME/Documents/PDF/"
DEST_DIR="archive/testCollection"
# Number of files to select
NUM_FILES=10

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# List all files in the source directory, shuffle them, and select the first NUM_FILES files
find "$SOURCE_DIR" -type f -name "*.pdf" | shuf | head -n "$NUM_FILES" | while read -r file; do
    # Copy each selected file to the current working directory
    cp "$file" $DEST_DIR
    echo "Copied: $file"
done

echo "$NUM_FILES files have been copied to the current directory."

