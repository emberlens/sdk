# Publishing the SDK

These steps are for the owner. Agents only prepare the text. Do them in this order: the notebook (step 3) installs from PyPI, so PyPI (step 2) comes first.

## 1. Make the public repo (owner)

1. On GitHub, create the empty public repo `emberlens/sdk`. Add no README, no licence and no .gitignore: the repo must stay empty, or the push below fails. The licence comes from the pushed `LICENSE` file.
2. In the main Emberlens repo, split out the `sdk/` folder:

   ```bash
   git subtree split --prefix=sdk -b sdk-public
   ```

3. Push that branch as `main` of the new repo:

   ```bash
   git push git@github.com:emberlens/sdk.git sdk-public:main
   ```

4. To update later, run the same two commands (`split`, then `push`).

## 2. Publish to PyPI (owner)

1. Check that the name `emberlens-sdk` is free on https://pypi.org/project/emberlens-sdk/ (it must show "Not Found").
2. Build from the `sdk/python` folder:

   ```bash
   cd sdk/python
   python3 -m pip wheel --no-deps -w dist .
   ```

   (Or `python3 -m build`, if you have it.)
3. Upload to TestPyPI first:

   ```bash
   python3 -m pip install twine
   python3 -m twine upload --repository testpypi dist/*
   ```

4. Install it in a fresh venv and check it:

   ```bash
   python3 -m venv /tmp/emberlens-check
   /tmp/emberlens-check/bin/pip install --index-url https://test.pypi.org/simple/ emberlens-sdk
   /tmp/emberlens-check/bin/emberlens selftest
   /tmp/emberlens-check/bin/emberlens --version
   ```

5. If both pass, upload to PyPI:

   ```bash
   python3 -m twine upload dist/*
   ```

## 3. Test the notebook (owner)

Do this after step 2.

1. Zip the sample export: `sdk/fixtures/coco_joined`.
2. Put the zip in Google Drive.
3. Open the notebook: https://colab.research.google.com/github/emberlens/sdk/blob/main/notebooks/train.ipynb
4. Paste the Drive path of the zip into the notebook, then choose **Run all**.
5. Do it twice: once with `classify` and once with `detect`.

Pass: a `.onnx` file appears next to the zip, and the run shows no error.

## 4. Turn the connector on (owner)

1. In `mcp/wrangler.toml` set:

   ```toml
   NOTEBOOK_REPO = "emberlens/sdk"
   SDK_PUBLISHED = "true"
   DETECT_EXPORT = "true"
   ```

2. Deploy as described in `mcp/README.md`.

## 5. Checklist before each release

- [ ] The versions match (a test checks this).
- [ ] `sdk/emberlens selftest` is green.
- [ ] No file outside `sdk/` is linked.
