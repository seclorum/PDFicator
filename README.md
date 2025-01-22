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

