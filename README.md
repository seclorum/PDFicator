# PDFicator

A hacky workspace for developing workflows to analyze a massive (+80k) collection of PDF documents.

The idea is, give the tool a massive directory of PDF files, and extract data from it.

This project will start off with a few hacky python experiments, but the eventual goal is to build an app that is worthy of the collection.  

Different approaches will be taken towards the goal of mining the data - static analysis, extraction, and eventual use in ML is intended.

 
## Project Structure:

```
.
├── LICENSE
├── README.md
├── archive
│   └── testCollection
├── data
├── pdf_tool_env
├── py
│   ├── process_pdfs.py
│   └── query_pdfs.py
├── process.sh
├── query.sh
└── tools.sh
```

## Notes:

* tools.sh is used to set up the tooling for the project, including a pyenv environment.
* For now, requires python 3.11 on MacOS and Linux.  Other python versions may get a pyenv configuration when needed.
* The archive/testCollection/ and data/ directories are .gitignor'ed!  
* Put your own test PDF collection in archive/testCollection/
* process.sh/query.sh will depend on the contents of data/
* Do not commit your data to the repo - be sure to inspect .gitignore 
* At least rustc>1.7 is required to be onboard, for the tokenizer packages installed by pip 

## Run:

To do a test run, first you must populate the archive/testCollection/ directory with some PDF's of your own.  You can use the ```populate.sh``` script to do this - it will copy 50 random files from the directory of your choice.

Once you've put some files in archive/testCollection/, you can run the following:

``` sh process.sh && sh query.sh ```

This will process the PDF's in the archive, and then give you a REPL-like query interface to search for text in the database created by process.sh.

Note that, at the moment, there are bugs in the indexing yet to be fixed - but limited results can be attained with this workflow so far.

