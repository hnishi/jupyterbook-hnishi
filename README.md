# jupyterbook-hnishi

Notebooks published by hnishi

## Usage

### Building the book

If you'd like to develop on and build the jupyterbook-hnishi book, you should:

- Clone this repository and run
- Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
- (Recommended) Remove the existing `jupyterbook-hnishi/_build/` directory
- Run `jupyter-book build jupyterbook-hnishi/`

A fully-rendered HTML version of the book will be built in `jupyterbook-hnishi/_build/html/`.

### Hosting the book

The html version of the book is hosted on the `gh-pages` branch of this repo. A GitHub actions workflow has been created that automatically builds and pushes the book to this branch on a push or pull request to main.

If you wish to disable this automation, you may remove the GitHub actions workflow and build the book manually by:

- Navigating to your local build; and running,
- `ghp-import -n -p -f jupyterbook-hnishi/_build/html`

This will automatically push your build to the `gh-pages` branch. More information on this hosting process can be found [here](https://jupyterbook.org/publish/gh-pages.html#manually-host-your-book-with-github-pages).

## Development

Install required packages in venv.

```shell
poetry install --no-root
```

Build html.

```shell
poetry run jupyter-book build jupyterbook_hnishi/
```

When you copy the notebook created by colab, you have to clear widget state to prevent the following error during build (idk why).

```shell
Traceback (most recent call last):
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/jupyter_book/sphinx.py", line 150, in build_sphinx
    app.build(force_all, filenames)
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/sphinx/application.py", line 352, in build
    self.builder.build_update()
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/sphinx/builders/__init__.py", line 296, in build_update
    self.build(to_build,
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/sphinx/builders/__init__.py", line 310, in build
    updated_docnames = set(self.read())
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/sphinx/builders/__init__.py", line 417, in read
    self._read_serial(docnames)
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/sphinx/builders/__init__.py", line 438, in _read_serial
    self.read_doc(docname)
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/sphinx/builders/__init__.py", line 478, in read_doc
    doctree = read_doc(self.app, self.env, self.env.doc2path(docname))
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/sphinx/io.py", line 221, in read_doc
    pub.publish()
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/docutils/core.py", line 217, in publish
    self.document = self.reader.read(self.source, self.parser,
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/sphinx/io.py", line 126, in read
    self.parse()
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/docutils/readers/__init__.py", line 77, in parse
    self.parser.parse(self.input, document)
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/myst_nb/parser.py", line 81, in parse
    md_parser, env, tokens = nb_to_tokens(
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/myst_nb/parser.py", line 203, in nb_to_tokens
    if contains_widgets(ntbk):
  File "/Users/hnishi/work/jupyterbook-hnishi/.venv/lib/python3.9/site-packages/jupyter_sphinx/execute.py", line 283, in contains_widgets
    return widgets and widgets["state"]
KeyError: 'state'
```

As far as I know, to clear widget states, open the notebook in your local jupyter notebook and choose the widget tab and click "clear widget state" and save the ipynb file.

When you push the updated codes into the main branch, GitHub Actions builds the html pages and update your site.

### Editing notebooks with Colaboratory

- Open a notebook
- Click the link to Colaboratory (in the top cell)
- Edit the notebook with Colaboratory
- Open a command bar (`shift` + `ctrl` + `p`)
- Type `Save a copy in GitHub`
- Select the `main` branch
- Check `include a link to Colaboratory`
- Click `OK`

### Deployment

Deployment is automatically done using GitHub Actions.

Please refer to .github/workflows/deploy.yml for more details.

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/hnishi/jupyterbook_hnishi/graphs/contributors).

## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).
