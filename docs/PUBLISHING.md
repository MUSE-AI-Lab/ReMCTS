# Publishing ReMCTS

## Recommended Hosting

- **GitHub:** source code, tests, small configuration files, documentation, and
  compact aggregate result tables.
- **GitHub Releases + Zenodo:** immutable versioned source archive and DOI for
  the exact paper artifact.
- **Hugging Face Dataset (optional):** only original or legally redistributable
  datasets/trajectories, with a dataset card and a separate license.

Do not put multi-gigabyte run directories into Git or Git LFS merely because
they exist locally. Publish a reviewed artifact archive only when it materially
supports reproduction.

## Before the First Commit

1. Confirm the MIT license with all authors and Shenzhen University.
2. Replace `YOUR_GITHUB_ACCOUNT` in `README.md` after choosing the owner.
3. Check that `my_api_key.txt`, `apikeys.txt`, `outputs/`, paper drafts, review
   material, and local datasets are ignored.
4. Rotate any credential that has ever been pasted into a message, log, commit,
   shared archive, or cloud drive.
5. Run the tests, package build, and secret scan described below.

## Create the GitHub Repository

Create an empty **private** repository named `ReMCTS`, preferably under the
authors' or lab's GitHub organization. Push the reviewed first commit, let CI
pass, inspect the GitHub file list, and only then change the repository to
public. Do not ask GitHub to generate a README, `.gitignore`, or license because
this directory already contains them.

With GitHub CLI:

```bash
git add .
git status --short
git commit -m "Release ReMCTS v0.1.0"
gh repo create ReMCTS --private --source=. --remote=origin --push
```

Without GitHub CLI, create the empty private repository in the browser and run
the remote commands shown by GitHub. After CI and the final file review pass,
change visibility to public in the repository settings.

## Archive a Version and Obtain a DOI

1. Connect the GitHub repository in Zenodo and enable automatic archiving.
2. Create and push the release tag:

```bash
git tag -a v0.1.0 -m "ReMCTS v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --generate-notes
```

3. Confirm that Zenodo archived the release and issued a version DOI.
4. Add the Zenodo DOI and final GitHub URL to `CITATION.cff`, then make a small
   metadata follow-up release if necessary.
5. Use the Zenodo concept DOI in the paper when you want one citation to resolve
   to the latest archived version; use the version DOI for exact reproducibility.

## Local Release Checks

```bash
pytest -q
python -m pip wheel . --no-deps --wheel-dir /tmp/remcts-wheel
git status --short --ignored
git ls-files --others --exclude-standard
```

Before pushing, inspect the complete staged set with `git diff --cached --stat`
and run a dedicated secret scanner such as Gitleaks if available. Never rely on
`.gitignore` alone after a secret has been committed; remove it from history and
rotate it.
