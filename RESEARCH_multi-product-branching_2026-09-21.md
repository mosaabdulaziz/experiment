# Research: managing ONE shared DeepStream pipeline across several products in one git repo

**Date:** 2026-09-21
**Scope:** branching models, release trains, DeepStream per-application configuration, safe merge-to-main practices, and mechanics for integrating several long-lived divergent branches.
**Method / trust rule:** every claim below is taken from a primary source — the organisation that owns the practice (Google DORA, martinfowler.com, Git's own documentation and Pro Git, GitLab's own docs source, GitHub Docs, NVIDIA DeepStream documentation). No secondary blog summaries were used. Each claim carries the URL it came from. Quoted text is quoted from the cited page.
**Note on placement:** this repo has no committed `docs/` directory (`docs/` is listed in `.gitignore`) and no pre-existing research/notes convention — the top-level Markdown files are handoff/architecture notes (`ADJUSTMENTS_2026-09-12.md`, `HANDOFF_2026-09-10.md`, `JUST_GUIDE.md`, `ARCHITECTURE.md`, `BOTTLENECKS.md`, `FILES.md`, `README.md`). This file therefore follows the existing `TOPIC_date.md` top-level pattern.

---

## 1. Branching models: trunk-based development vs long-lived feature branches

### What DORA (Google's DevOps Research and Assessment) actually claims

DORA's trunk-based development capability page states the practice concretely:

- "Branches in trunk-based development typically last no more than a few hours, with many developers merging their individual changes into trunk frequently."
- "Have three or fewer active branches in the application's code repository."
- "Merge branches to trunk at least once a day."
- On long-lived branches: "These changes require bigger and more complex merge events when compared to trunk-based development", and "large merges frequently introduce bugs or regressions."
- The claimed effect: "Teams achieve higher levels of software delivery and operational performance (delivery speed, stability, and availability) if they follow these practices."

Source: https://dora.dev/capabilities/trunk-based-development/

DORA's continuous integration page adds the supporting mechanics and the claimed outcome:

- CI is where "developers integrate all their work into the main version of the code base (known as trunk, main, or mainline) on a regular basis."
- "Each commit should trigger a build of the software." and "Each commit should trigger a series of automated tests that provide feedback in a few minutes."
- The stated benefit: CI "leads to higher deployment frequency, more stable systems, and higher quality software", and it ensures "the software is always in a working state, and that developer branches don't diverge significantly from trunk."

Source: https://dora.dev/capabilities/continuous-integration/

> Important precision: DORA's claim is about *branch count and merge cadence* correlating with delivery performance. It does **not** claim that a product-per-branch layout is impossible — it claims the merge events get bigger and riskier.

### What Martin Fowler's patterns actually claim

From "Patterns for Managing Source Code Branches":

- Integration is defined as a two-way act, not just pulling: "Developers integrate their work by pulling from mainline, merging, and - if healthy - pushing back into mainline" (Mainline Integration).
- The core frequency claim: "Frequent integration increases the frequency of merges but reduces their complexity and risk."
- Textual merge success is not correctness: "But often conflicts appear where the text merges without a problem, but the system still doesn't work... This is what I call a Semantic Conflict."
- Fowler reports the same DevOps finding: "branches or forks with very short lifetimes (less than a day) before being merged into trunk, and less than three active branches in total, are important aspects of continuous delivery."
- On feature branching's status: "Feature Branching appears to be the most common branching strategy in the industry at the moment, but there is a vocal group of practitioners who argue that Continuous Integration is usually a superior approach."

Source: https://martinfowler.com/articles/branching-patterns.html

The semantic-conflict point is the one that matters most for a shared CV pipeline: four product branches can each edit a shared perception/pipeline/config module, merge cleanly at the text level, and still produce a pipeline that mis-batches streams or double-registers a probe.

### What GitLab's own flow documentation claims

From GitLab's own source for the GitLab Flow document:

- "It is a convention to call your default branch `main` and to mostly branch from and merge to this."
- "Nowadays, most organizations practice continuous delivery, which means that your default branch can be deployed."
- "Most feature branches should take less than one day of work." and "If your feature branches often take more than a day of work, try to split your features into smaller units of work."
- On what goes wrong otherwise: "The biggest problem is that many long-running branches emerge that all contain part of the changes."
- The prescribed fix: "The last reason for creating merge commits is to keep long-running feature branches up-to-date with the latest state of the project. The solution here is to keep your feature branches short-lived."
- On conflicts: "If your feature branch has a merge conflict, creating a merge commit is a standard way of solving this."
- On rebasing shared branches: "However, you should avoid rebasing commits you have pushed to a remote server if you have other active contributors in the same branch." and "You should never rebase commits authored by other people unless you've agreed otherwise."

Source (GitLab's own repository, the canonical doc source): https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/topics/gitlab_flow.md

### What GitHub's own flow documentation claims

GitHub flow is documented as six steps: create a branch, make changes, create a pull request, address review comments, merge, delete the branch. Branches are explicitly disposable: "After you merge your pull request, delete your branch. This indicates that the work on the branch is complete and prevents you or others from accidentally using old branches." And on scoping: "Make a separate branch for each set of unrelated changes. This makes it easier for reviewers to give feedback."

Source: https://docs.github.com/en/get-started/using-github/github-flow

### What Git's own workflow documentation claims

`gitworkflows(7)` is Git's own prescriptive document, and it is notably *not* trunk-only — it documents a graduation model with several integration branches:

- **Rule: Topic branches** — "Make a side branch for every topic (feature, bugfix, ...). Fork it off at the oldest integration branch that you will eventually want to merge it into."
- **Rule: Merge upwards** — "Always commit your fixes to the oldest supported branch that requires them. Then (periodically) merge the integration branches upwards into each other."
- Integration branches in git.git: *maint* (next maintenance release), *master* (next feature release), *next* (testing for stability), *seen* (not quite ready). "Each of the four branches is usually a direct descendant of the one above it. Conceptually, the feature enters at an unstable branch (usually *next* or *seen*), and 'graduates' to *master* for the next release once it is considered stable enough."
- **Rule: Merge to downstream only at well-defined points** — "Do not merge to downstream except with a good reason: upstream API changes affect your branch; your branch no longer merges to upstream cleanly; etc." The reason given: "Otherwise, the topic that was merged to suddenly contains more than a single (well-separated) change. The many resulting small merges will greatly clutter up history."
- Constraint that bites long-lived branches: "Note that the last point clashes with the other two: a topic that has been merged elsewhere should not be rebased."

Source: https://git-scm.com/docs/gitworkflows

**Net of section 1 (sourced):** the primary sources agree that merge cost and defect risk scale with divergence (DORA, Fowler) and that the remedy is frequent mainline integration plus short branch lifetime (DORA, Fowler, GitLab, GitHub). Git's own workflow doc shows the legitimate exception — *integration* branches that exist permanently and are merged **upwards** on a schedule — which is different from long-lived *feature* branches.

---

## 2. Release branches / release trains: "a frozen release per customer" while development continues

### GitLab Flow's documented release-branch rules

- "Create stable branches using `main` as a starting point, and branch as late as possible. By doing this, you minimize the length of time during which you have to apply bug fixes to multiple branches."
- "After announcing a release branch, only add serious bug fixes to the branch."
- **Upstream-first is explicit:** "If possible, first merge these bug fixes into `main`, and then cherry-pick them into the release branch." GitLab names this policy and its rationale: merging into the release branch first and forgetting to bring the fix back to `main` reproduces the same bug in the next release.
- "Every time you include a bug fix in a release branch, increase the patch version (to comply with Semantic Versioning) by setting a new tag."
- "Some projects also have a stable branch that points to the same commit as the latest released branch."

Source: https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/topics/gitlab_flow.md

### GitLab Flow's environment branches (the "frozen deployment" variant)

GitLab documents a second, distinct pattern for environments rather than versions:

- "You can deploy a new version by merging `main` into the `production` branch."
- "If you need to know what code is in production, you can check out the production branch to see."
- "In this case, deploy the `staging` branch to your staging environment. To deploy to pre-production, create a merge request from the `staging` branch to the `pre-prod` branch. Go live by merging the `pre-prod` branch into the `production` branch. This workflow, where commits only flow downstream, ensures that everything is tested in all environments."

Source: https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/topics/gitlab_flow.md

This is the directly relevant pattern for "a frozen deployment per customer": a **customer/site branch is an environment branch, not a feature branch** — code only ever flows *downstream* into it from `main`, never the reverse.

### Fowler's Release Branch pattern

- Definition: a release branch is "A branch that only accepts commits accepted to stabilize a version of the product ready for release."
- Fowler's guidance is the same upstream-first rule: fixes are made on mainline first and then cherry-picked into the release branch, to prevent the fix regressing in the next release.
- Fowler also flags the scaling limit of the "Long Lived Release Branch" variation: maintaining multiple simultaneously-supported release branches gets harder over time as they diverge.

Source: https://martinfowler.com/articles/branching-patterns.html

### Git's own equivalent

`gitworkflows(7)`'s "Merge upwards" rule is the same policy expressed for a maintenance branch: commit the fix to the *oldest* supported branch that needs it, then merge integration branches upward; cherry-picking downward is described as the rare exception.

Source: https://git-scm.com/docs/gitworkflows

**Net of section 2 (sourced):** every first-party flow that documents frozen releases uses the same two rules — (a) branch as late as possible, (b) fix on `main` first and cherry-pick into the frozen branch, not the other way round. Customer-specific *behaviour* is not supposed to live on the frozen branch as unique code; the frozen branch is supposed to be a stabilised snapshot plus cherry-picked fixes.

---

## 3. Monorepo with per-product modules vs plugin/extension split: what NVIDIA's DeepStream docs actually support

NVIDIA's documentation is explicit that per-application behaviour belongs in **configuration files and dynamically loaded custom libraries**, not in edits to shared pipeline code.

### 3.1 The reference application is configured entirely by a config file

The DeepStream reference application takes its whole pipeline shape from one key-file-format config passed as `-c config_file`. The documented groups include `[application]`, `[source0]`/`[source1]`..., `[streammux]`, `[pre-process]`, `[primary-gie]`, `[secondary-gie0]`/`[secondary-gie1]`, `[tracker]`, `[tiled-display]`, `[osd]`, `[sink0]`/`[sink1]`, `[nvds-analytics]`, `[tests]`. The documented purposes of the config file are to "Enable or disable components", "Change the properties or behavior of components", and "Customize other application configuration settings".

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html

**Implication supported by the doc:** a product variant is expressible as *a different config file over the same binary*, which is a file no other product needs to touch.

### 3.2 Per-model inference config is a separate file per GIE

`Gst-nvinfer` takes `config-file-path`: "Absolute pathname of configuration file for the Gst-nvinfer element." Its config file has the groups `[property]` (the only mandatory group), `[class-attrs-all]`, and `[class-attrs-<class-id>]` (e.g. `[class-attrs-23]`) for per-class detection parameters.

Identity and chaining properties let several models coexist in one pipeline without interfering:

- `gie-unique-id`: "Unique ID to be assigned to the GIE to enable the application and other elements to identify detected bounding boxes and labels"
- `operate-on-gie-id`: "Unique ID of the GIE on whose metadata (bounding boxes) this GIE is to operate on"
- `operate-on-class-ids`: "Class IDs of the parent GIE on which this GIE is to operate on" (semicolon-delimited)
- `process-mode`: "Integer 1=Primary 2=Secondary" — primary "Operates on full frames", secondary "Operates on objects added in the meta by upstream components".

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html

**Implication supported by the doc:** fire/smoke detection and face/attendance detection can be two GIEs with distinct `gie-unique-id`s and distinct nvinfer config files; downstream consumers select by `gie-unique-id` rather than by a code branch.

### 3.3 Custom output parsers are separate shared libraries, loaded by name

NVIDIA documents the custom-parser contract as a **dlopen'd library plus a symbol name in the config**:

- "All the interface implementations for the models must go into a single independent shared library."
- "nvinfer dynamically loads the library with `dlopen()`, looks for implemented interfaces with `dlsym()`, and calls the interfaces as required."
- For detectors, "you must write a library that can parse the bounding box coordinates and the object class from the output layers"; for classifiers, "the library must parse the object attributes from the output layers."
- Config keys: `custom-lib-path` (library pathname), `parse-bbox-func-name` (detector parser), `parse-classifier-func-name` (classifier parser), `engine-create-func-name` (CUDA engine creation).

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_using_custom_model.html

The API reference pins the contract:

- "The custom parsing function can be specified in the Gst-nvinfer configuration file by the properties `parse-bbox-func-name` (name of the parsing function) and `custom-lib-name`."
- "The Gst-nvinfer plugin loads the library and looks for the custom parsing function symbol. The function is called after each inference call is executed."
- Typedefs: `NvDsInferParseCustomFunc`, `NvDsInferClassiferParseCustomFunc`, `NvDsInferInstanceMaskParseCustomFunc`; the `CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE` macro "Must be called after defining the function."

Source: https://docs.nvidia.com/metropolis/deepstream/7.1/sdk-api/nvdsinfer__custom__impl_8h.html

**Implication supported by the doc:** a per-product parser (e.g. a fire/smoke YOLO head) is a *new symbol in a new .so referenced from that product's config* — it does not require editing any shared parser used by another product.

### 3.4 Analytics rules are per-stream, keyed by stream id

`Gst-nvdsanalytics` takes a `config-file` — "Configuration file which consist of the analytics rules" — whose groups are `[property]` (the only mandatory group) plus per-stream groups `[roi-filtering-stream-<n>]`, `[overcrowding-stream-<n>]`, `[direction-detection-stream-<n>]`, `[line-crossing-stream-<n>]`, where `<n>` is the stream ID (e.g. `[roi-filtering-stream-0]` configures ROI filtering for stream ID 0).

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsanalytics.html

**Implication supported by the doc:** per-customer ROI/line/zone behaviour is addressable per stream index in a config file. (This repo already generates `data/nvdsanalytics_config.txt` and per-process remapped copies at runtime and gitignores them, which matches the documented model.)

### 3.5 Custom algorithms: NVIDIA's two documented extension points

**(a) Template plugin with a hooked custom library (no pipeline edit):**
"The OSS Gst-nvdsvideotemplate plugin is a template plugin which provides a custom library hooking interface for single/batched video frame(s) processing." Its properties are `customlib-name` ("custom library Name to be used") and `customlib-props` ("Custom Library Properties (key:value) string, can be set multiple times, vector is maintained internally"). The custom library implements `SetInitParams`, `HandleEvent`, `GetCompatibleCaps`, `ProcessBuffer`, `SetProperty`.

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsvideotemplate.html

**(b) The dsexample route:**
"The DeepStream SDK supports a mechanism to add third party or custom algorithms within the reference application by modifying the example plugin (`gst-dsexample`)." Sources live in `sources/gst-plugins/gst-dsexample`. "The GStreamer plugin itself is a standard in-place transform plugin. Because it does not generate new buffers but only adds / updates existing metadata, the plugin implements an in-place transform." The static library `dsexample_lib` demonstrates the plugin↔custom-library interface via `DsExampleCtxInit`, `DsExampleCtxDeinit`, `DsExampleProcess`.

Source: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_sample_custom_gstream.html

**Net of section 3 (sourced):** NVIDIA's documented architecture gives four product-isolation seams that need no branch at all — (1) per-application deepstream-app config file, (2) per-model nvinfer config + `gie-unique-id`/`operate-on-gie-id`, (3) per-model custom parser in its own `.so` selected by `custom-lib-path` + `parse-bbox-func-name`, (4) per-stream analytics groups. The pattern NVIDIA documents as *requiring* shared-source edits is (b) above — modifying `gst-dsexample` — which is exactly the kind of change that needs a plugin seam (a) instead if two products would both touch it.

---

## 4. Safe merge-to-main practices that stop one product breaking another

### 4.1 Feature flags / toggles (Fowler's canonical article)

- Release Toggles are the mechanism that makes trunk integration of unfinished work possible: "Release Toggles allow incomplete and un-tested codepaths to be shipped to production."
- The article names the exact problem this repo has: "You want to avoid branching for this work if at all possible, based on previous painful experiences of merging long-lived branches."
- Four documented categories: **Release Toggles**, **Experiment Toggles**, **Ops Toggles** (operational control / graceful degradation without redeployment), **Permissioning Toggles** (feature restricted to a specific set of users — the closest documented analogue to per-customer enablement; "a Champagne Brunch feature is exposed to a specific set of users").
- Two axes: **transient vs long-lived** (Release Toggles days/weeks; Permissioning Toggles may persist years) and **static vs dynamic**.
- Configuration approaches, simplest to most complex: hardcoded → parameterized (env vars / CLI args) → configuration files in source control → application database with admin UI → distributed config (Zookeeper/etcd/Consul). Fowler's preference: "Managing toggle configuration via source control and re-deployments is preferable, if the nature of the feature flag allows it."
- Toggle debt is real and must be managed: "Savvy teams view the Feature Toggles in their codebase as inventory which comes with a carrying cost and seek to keep that inventory as low as possible." Documented removal tactics: add a removal task to the backlog when the toggle is introduced, set expiry dates, add "time bombs" that fail when a toggle outlives its date, cap the total number of toggles.

Source: https://martinfowler.com/articles/feature-toggles.html

Fowler's branching-patterns article lists the same three trunk-compatible techniques for partially-complete work: **Keystone Interface** (add the UI last), **Feature Flag**, **Branch By Abstraction**.
Source: https://martinfowler.com/articles/branching-patterns.html

### 4.2 CI gates on the protected branch

GitHub's protected-branch documentation states:

- "Repository administrators or custom roles with the 'edit repository rules' permission can require that all pull requests receive a specific number of approving reviews before someone merges the pull request into a protected branch."
- "If you enable required reviews, collaborators can only push changes to a protected branch via a pull request that is approved by the required number of reviewers with write permissions."
- "Required status checks must have a `successful`, `skipped`, or `neutral` status before collaborators can make changes to a protected branch."
- "Strict" required status checks mean "The branch **must** be up to date with the base branch before merging."
- Require linear history "prevents collaborators from pushing merge commits to the branch."

Source: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches

DORA's CI requirements give the content of those gates: a build on every commit and "a series of automated tests that provide feedback in a few minutes."
Source: https://dora.dev/capabilities/continuous-integration/

### 4.3 CODEOWNERS on the shared core

**GitHub:** "Code owners are automatically requested for review when someone opens a pull request that modifies code that they own." The file is searched for in `.github/`, the repository root, or `docs/`, and "GitHub will search for them in that order and use the first one it finds"; "you can assign different code owners for different branches". Precedence: "Order is important; the last matching pattern takes the most precedence." With branch protection, "Repository owners can update branch protection rules to ensure that changed code is reviewed by the owners of the changed files", and "an approval from *any* of the owners is sufficient to meet this requirement". Limit: "CODEOWNERS files must be under 3 MB in size."
Source: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners

**GitLab (relevant here, since this repo is GitLab-hosted):** Code Owners "Define the owners of files and directories in a repository" and "Require owners to approve changes. Combine protected branches with Code Owners to require experts to approve merge requests." GitLab uses a single CODEOWNERS file and checks, in order: `./CODEOWNERS`, `./docs/CODEOWNERS`, `./.gitlab/CODEOWNERS`. Enforcement requires the target branch to be protected with Code Owner approval enabled; users with **Allowed to push and merge** can bypass the merge request process entirely.
Source: https://docs.gitlab.com/user/project/codeowners/

> Caveat for this repo: `docs/` is gitignored here, so a CODEOWNERS file must go at `./CODEOWNERS` or `./.gitlab/CODEOWNERS`.

### 4.4 Merge queues / merge trains — the defence against "both PRs were green, main is red"

**GitLab merge trains** put merge requests "in a queue" where "each merge request is compared to the other, earlier merge requests, to ensure they all work together." The failure mode they exist to prevent is stated directly: "Two merge requests can each pass their own pipeline, but their combined changes can still conflict. If both merge, the target branch can break, even though every pipeline succeeded." A merged-results pipeline tests "one merge request's changes combined with the target branch", but it "does not account for other merge requests that merge around the same time" — the train tests each request "against the combined changes of every merge request ahead of it in the queue. This catches conflicts before they reach the target branch."
Source: https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/ci/pipelines/merge_trains.md (rendered: https://docs.gitlab.com/ci/pipelines/merge_trains/)

**GitHub merge queue:** "A merge queue helps increase velocity by automating pull request merges into a busy branch and ensuring the branch is never broken by incompatible changes." Queued PRs are "grouped into a `merge_group` with the latest version of the `base_branch` as well as changes from pull requests ahead of it in the queue"; "GitHub will merge all these changes into the `base_branch` once the checks required by the branch protections of `base_branch` pass"; a failing group entry is evicted automatically. Workflows must be configured to trigger on merge-group events.
Source: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue

### 4.5 Per-product smoke tests scoped by changed paths

**GitLab:** `rules:changes` runs a job conditionally on modified files; `rules:changes:paths` names the files to watch; the branch is compared against a reference (`rules:changes:compare_to`), and if there are no changed files matching, the job does not run. `rules:changes` combines with `if:` and, for `parallel:matrix` jobs, matrix variables can be used in `rules:changes` paths "to run each job instance only when files relevant to that matrix value have changed."
Source: https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/ci/jobs/job_rules.md (rendered: https://docs.gitlab.com/ci/jobs/job_rules/)

**GitHub Actions:** "When using the `push` and `pull_request` events, you can configure a workflow to run based on what file paths are changed." A workflow runs when at least one changed file matches `paths`; "When all the path names match patterns in `paths-ignore`, the workflow will not run." `paths` and `paths-ignore` cannot both be used for the same event; negation with `!` inside `paths` provides the combination, and "The order that you define `paths` patterns matters: A matching negative pattern (prefixed with `!`) after a positive match will exclude the path."
Source: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax

> Precision worth keeping: path-scoped jobs are an *efficiency* mechanism. A change to the shared DeepStream sidecar matches every product's path filter, so the shared core must be a path that triggers **all** product smoke tests, not a per-product one.

---

## 5. Practical mechanics for this situation: several long-lived branches that each edited shared core files

Primary-source mechanics only. The repo situation assumed: `feat/fire-smoke-attendance` (~75 commits ahead of `main`, touching shared `app/` and `perception/pipeline/`), `feat/barber-shop-monitoring`, `feat/cognitive-monitoring`, plus ~25 other branches.

### 5.1 Know the actual divergence first: `git merge-base`

- A merge base is the best common ancestor: "One common ancestor is *better* than another common ancestor if the latter is an ancestor of the former. A common ancestor that does not have any better common ancestor is a *best common ancestor*, i.e. a *merge base*."
- `git merge-base --is-ancestor A B` "Check if the first <commit> is an ancestor of the second <commit>, and exit with status 0 if true, or with status 1 if not." — the scriptable way to ask "is this branch already fully contained in main?"
- `git merge-base --octopus` "Computes the best common ancestors of all supplied commits in preparation for an n-way merge."
- `git merge-base --independent` prints "a minimal subset of the supplied commits with the same ancestors. In other words, among the commits given, list those which cannot be reached from any other." — i.e. which of the ~28 branches are actually distinct tips worth merging.
- `--fork-point` "Find the point at which a branch (or any history that leads to <commit>) forked from another branch (or any reference) <ref>", and it "does not just look for the common ancestor of the two commits, but also takes into account the reflog of <ref>", which matters if `main` was ever rewound.
- Criss-cross histories give more than one best merge base: "When the history involves criss-cross merges, there can be more than one *best* common ancestor for two commits... Neither one is better than the other (both are *best* merge bases)." Use `-a|--all` to see them all.

Source: https://git-scm.com/docs/git-merge-base

### 5.2 Choose the merge strategy deliberately

From the `git merge` manual:

- **`ort` is the default** for merging one branch. "This strategy can only resolve two heads using a 3-way merge algorithm. When there is more than one common ancestor that can be used for 3-way merge, it creates a merged tree of the common ancestors and uses that as the reference tree for the 3-way merge. This has been reported to result in fewer merge conflicts without causing mismerges." It also "can detect and handle merges involving renames."
- `recursive` is now a synonym for `ort` ("It was an alternative implementation until v2.49.0, but was redirected to mean `ort` in v2.50.0").
- **`octopus`** "resolves cases with more than two heads, but refuses to do a complex merge that needs manual resolution. It is primarily meant to be used for bundling topic branch heads together." → **an octopus merge of four divergent product branches that touched the same shared files will simply refuse.** Merging them one at a time is the documented path.
- **`ours` (strategy)** "resolves any number of heads, but the resulting tree of the merge is always that of the current branch head, effectively ignoring all changes from all other branches." This is the "record the merge, keep none of their content" tool — the doc warns explicitly it "should not be confused with" `-X ours`, which "does not even look at what the other tree contains at all."
- **`-X ours` / `-X theirs` (strategy options)**: `-X ours` "forces conflicting hunks to be auto-resolved cleanly by favoring *our* version. Changes from the other tree that do not conflict with our side are reflected in the merge result." `-X theirs` "is the opposite of `ours`".
- **`-X renormalize`** "runs a virtual check-out and check-in of all three stages of any file which needs a three-way merge... meant to be used when merging branches with different clean filters or end-of-line normalization rules."
- **`-X ignore-space-change` / `ignore-all-space` / `ignore-space-at-eol` / `ignore-cr-at-eol`** "Treats lines with the indicated type of whitespace change as unchanged for the sake of a three-way merge" — with the documented tie-breaks: "If *their* version only introduces whitespace changes to a line, *our* version is used"; "If *our* version introduces whitespace changes but *their* version includes a substantial change, *their* version is used."
- **`-X find-renames[=<n>]`** "Turn on rename detection, optionally setting the similarity threshold. This is the default"; `no-renames` turns it off. Relevant when a product branch moved a shared module.
- **`-X diff-algorithm=(histogram|minimal|myers|patience)`** "Use a different diff algorithm while merging, which can help avoid mismerges that occur due to unimportant matching lines (such as braces from distinct functions). Note that `ort` defaults to `diff-algorithm=histogram`."
- Conflict presentation: default RCS-style markers; **`diff3`** adds the `|||||||` base section; **`zdiff3`** "is similar to diff3 but removes matching lines on the two sides from the conflict region when those matching lines appear near either the beginning or end of a conflict region." Set via `merge.conflictStyle`.

Source: https://git-scm.com/docs/git-merge

### 5.3 Don't re-solve the same conflict four times: `git rerere`

Git's own manual describes precisely this scenario:

- "In a workflow employing relatively long lived topic branches, the developer sometimes needs to resolve the same conflicts over and over again until the topic branches are done (either merged to the 'release' branch, or sent out and accepted upstream)."
- "You need to set the configuration variable `rerere.enabled` in order to enable this command."
- Once enabled it is automatic: "*git merge* automatically invokes *git rerere* upon exiting with a failed automerge and *git rerere* records the hand resolve when it is a new conflict, or reuses the earlier hand resolve when it is not. *git commit* also invokes *git rerere* when committing a merge result." Result: "you do not have to do anything special yourself (besides enabling the rerere.enabled config variable)."
- Important limit: "Note that *git rerere* leaves the index file alone, so you still need to do the final sanity checks with `git diff` (or `git diff -c`) and *git add* when you are satisfied."

Source: https://git-scm.com/docs/git-rerere

Pro Git (git-scm.com's own book) lists the matching use cases: "when you want to make sure a long-lived topic branch will ultimately merge cleanly, but you don't want to have a bunch of intermediate merge commits cluttering up your commit history"; keeping a branch rebased "so you don't have to deal with the same rebasing conflicts each time"; and throw-away test merges — "you can rewind the merges and re-do them without the topic branch that made the tests fail without having to re-resolve the conflicts again." Enable with `git config --global rerere.enabled true`; inspect with `git rerere status` and `git rerere diff`.

Source: https://git-scm.com/book/en/v2/Git-Tools-Rerere

### 5.4 Rehearse the combination on a throw-away branch (Git's own documented rule)

`gitworkflows(7)`, **Rule: Throw-away integration branches**: "To test the interaction of several topics, merge them into a throw-away branch. You must never base any work on such a branch!" And: "If you make it (very) clear that this branch is going to be deleted right after the testing, you can even publish this branch, for example to give the testers a chance to work with it, or other developers a chance to see if their in-progress work will be compatible." git.git itself maintains *seen* as exactly this.

Source: https://git-scm.com/docs/gitworkflows

With `rerere.enabled` on, conflict resolutions discovered on the throw-away integration branch are replayed automatically when the same merges are done for real — this is the documented pairing of 5.3 and 5.4.

### 5.5 Tools for resolving the conflicts themselves (Pro Git, "Advanced Merging")

- Restart cleanly: "The `git merge --abort` option tries to revert back to your state before you ran the merge."
- More context in markers: `git checkout --conflict=diff3 <file>` "will use a slightly different version of conflict markers, not only giving you the 'ours' and 'theirs' versions, but also the 'base' version inline to give you more context." Persist with `git config --global merge.conflictstyle diff3`.
- Extract all three stages of a conflicted file for manual re-merge: `git show :1:file` (common ancestor), `git show :2:file` (ours), `git show :3:file` (theirs), then `git merge-file -p ours.rb common.rb theirs.rb > file`.
- Find *why* it conflicts: `git log --oneline --left-right --merge` shows the commits on each side that touched the conflicted file; "If you run that with the `-p` option instead, you get just the diffs to the file that ended up in conflict."
- Whitespace-only noise: `git merge -Xignore-space-change <branch>`.
- Side preference: `git merge -Xours <branch>` / `git merge -Xtheirs <branch>`; full override `git merge -s ours <branch>` — "This will simply record as the result of the merge the exact code in your current branch."
- Backing out an already-merged merge on a shared branch: `git revert -m 1 HEAD`, where "The `-m 1` flag indicates which parent is the 'mainline' and should be kept." Pro Git notes `git reset --hard` is "rewriting history, which can be problematic with a shared repository."

Source: https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging

### 5.6 The rebase trap for already-shared branches

`gitworkflows(7)` states the constraint plainly: "a topic that has been merged elsewhere should not be rebased." GitLab Flow says the same for shared feature branches: "you should avoid rebasing commits you have pushed to a remote server if you have other active contributors in the same branch" and "You should never rebase commits authored by other people unless you've agreed otherwise."

Sources: https://git-scm.com/docs/gitworkflows and https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/topics/gitlab_flow.md

**Net of section 5 (sourced):** the documented sequence for merging N divergent long-lived branches is: enable `rerere.enabled`; use `git merge-base --independent` / `--is-ancestor` to find which tips are genuinely distinct; build a throw-away integration branch and merge the product branches into it **one at a time** (octopus is documented to refuse complex merges); resolve conflicts once with `diff3`/`zdiff3` markers and the three-stage tools; delete the throw-away branch; then perform the real merges into `main`, where `rerere` replays the recorded resolutions. Use merge commits, not rebase, on branches other people have pushed to.

---

## What this implies for a shared-pipeline repo

Factual, sourced implications only — no recommendations beyond what the cited sources state.

1. **The merge cost the team is facing is the documented, expected outcome of the branch layout.** DORA states long-lived branches "require bigger and more complex merge events" and that "large merges frequently introduce bugs or regressions" (https://dora.dev/capabilities/trunk-based-development/); Fowler states "Frequent integration increases the frequency of merges but reduces their complexity and risk" (https://martinfowler.com/articles/branching-patterns.html); GitLab names the exact symptom: "The biggest problem is that many long-running branches emerge that all contain part of the changes" (https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/topics/gitlab_flow.md).

2. **A clean textual merge of four product branches over a shared pipeline is not evidence of a working pipeline.** Fowler: "often conflicts appear where the text merges without a problem, but the system still doesn't work... This is what I call a Semantic Conflict" (https://martinfowler.com/articles/branching-patterns.html). DORA's answer is a build plus automated tests on every commit giving "feedback in a few minutes" (https://dora.dev/capabilities/continuous-integration/).

3. **NVIDIA already documents product separation that needs no branch.** Per-application `deepstream-app` config groups (https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html); per-model `config-file-path` plus `gie-unique-id` / `operate-on-gie-id` / `operate-on-class-ids` (https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html); per-model custom parsers in their own shared library, since "All the interface implementations for the models must go into a single independent shared library" loaded via `dlopen()`/`dlsym()` and selected by `custom-lib-path` + `parse-bbox-func-name` (https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_using_custom_model.html); per-stream analytics rules keyed by stream id (https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsanalytics.html). Behaviour differences expressible in these four places do not require a per-product edit to a shared source file.

4. **Where a product genuinely needs new pipeline behaviour, NVIDIA documents a hooking seam rather than a shared-source edit.** `Gst-nvdsvideotemplate` "provides a custom library hooking interface" selected by `customlib-name` and configured by `customlib-props` (https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsvideotemplate.html). The alternative NVIDIA documents — "modifying the example plugin (`gst-dsexample`)" (https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_sample_custom_gstream.html) — is the one that puts two products in the same file.

5. **Unfinished or product-specific behaviour can be on `main` without being on for everyone.** Fowler: "Release Toggles allow incomplete and un-tested codepaths to be shipped to production", and the article names the motivation as wanting "to avoid branching for this work if at all possible, based on previous painful experiences of merging long-lived branches" (https://martinfowler.com/articles/feature-toggles.html). Per-customer enablement corresponds to his Permissioning Toggle category, which is documented as long-lived.

6. **Toggles are inventory with a cost, not a free lunch.** "Savvy teams view the Feature Toggles in their codebase as inventory which comes with a carrying cost and seek to keep that inventory as low as possible", with documented removal tactics including expiry dates and time bombs (https://martinfowler.com/articles/feature-toggles.html). Fowler also prefers static config in source control "if the nature of the feature flag allows it" — which matches DeepStream's own config-file model.

7. **A frozen per-customer release is a documented pattern, with a documented direction of flow.** GitLab: "Create stable branches using `main` as a starting point, and branch as late as possible"; "After announcing a release branch, only add serious bug fixes to the branch"; "first merge these bug fixes into `main`, and then cherry-pick them into the release branch" (https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/topics/gitlab_flow.md). Git's own rule is the same shape: "Always commit your fixes to the oldest supported branch that requires them. Then (periodically) merge the integration branches upwards into each other" (https://git-scm.com/docs/gitworkflows). Fowler notes the scaling limit: multiple simultaneously-maintained release branches get harder as they diverge (https://martinfowler.com/articles/branching-patterns.html).

8. **If per-customer deployments are environments rather than versions, GitLab documents a different shape:** commits flow only downstream through `staging` → `pre-prod` → `production`, and "This workflow, where commits only flow downstream, ensures that everything is tested in all environments" (https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/topics/gitlab_flow.md).

9. **Shared-core changes can be gated by ownership and by CI, and this repo is GitLab-hosted.** GitLab Code Owners requires a protected target branch with Code Owner approval enabled, uses a single CODEOWNERS file found at `./CODEOWNERS`, `./docs/CODEOWNERS` or `./.gitlab/CODEOWNERS`, and is bypassable by users with **Allowed to push and merge** (https://docs.gitlab.com/user/project/codeowners/). Note `docs/` is gitignored in this repo, so `./CODEOWNERS` or `./.gitlab/CODEOWNERS` is the viable location. GitHub's equivalent adds the precedence rule "the last matching pattern takes the most precedence" and "an approval from *any* of the owners is sufficient" (https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).

10. **Green-per-MR is not green-on-main when several product MRs land together.** GitLab states it directly: "Two merge requests can each pass their own pipeline, but their combined changes can still conflict. If both merge, the target branch can break, even though every pipeline succeeded" — merge trains test each MR "against the combined changes of every merge request ahead of it in the queue" (https://docs.gitlab.com/ci/pipelines/merge_trains/). GitHub's merge queue makes the same guarantee: it ensures "the branch is never broken by incompatible changes" (https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue).

11. **Per-product smoke tests can be path-scoped, but the shared core must trigger all of them.** GitLab's `rules:changes` runs a job only when matching files changed (https://docs.gitlab.com/ci/jobs/job_rules/); GitHub Actions' `paths`/`paths-ignore` do the same for workflows (https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax). Since a change to the shared DeepStream sidecar matches every product, the shared paths are the ones that must fan out to every product's smoke test.

12. **Git's own documentation supplies the mechanics for the one-off integration.** `octopus` "refuses to do a complex merge that needs manual resolution", so four conflicting product branches must be merged one at a time (https://git-scm.com/docs/git-merge); `git merge-base --independent` / `--is-ancestor` identify which of the ~28 branches are genuinely distinct (https://git-scm.com/docs/git-merge-base); throw-away integration branches are Git's documented way to test the interaction of several topics, with "You must never base any work on such a branch!" (https://git-scm.com/docs/gitworkflows); and `rerere` exists exactly because "In a workflow employing relatively long lived topic branches, the developer sometimes needs to resolve the same conflicts over and over again" (https://git-scm.com/docs/git-rerere).

13. **Do not rebase these branches.** "a topic that has been merged elsewhere should not be rebased" (https://git-scm.com/docs/gitworkflows); "you should avoid rebasing commits you have pushed to a remote server if you have other active contributors in the same branch" (https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/topics/gitlab_flow.md).

---

## Source list (all primary)

| Topic | URL |
|---|---|
| DORA — Trunk-based development | https://dora.dev/capabilities/trunk-based-development/ |
| DORA — Continuous integration | https://dora.dev/capabilities/continuous-integration/ |
| Fowler — Patterns for Managing Source Code Branches | https://martinfowler.com/articles/branching-patterns.html |
| Fowler — Feature Toggles (aka Feature Flags) | https://martinfowler.com/articles/feature-toggles.html |
| Git — gitworkflows(7) | https://git-scm.com/docs/gitworkflows |
| Git — git-merge | https://git-scm.com/docs/git-merge |
| Git — git-merge-base | https://git-scm.com/docs/git-merge-base |
| Git — git-rerere | https://git-scm.com/docs/git-rerere |
| Pro Git — Advanced Merging | https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging |
| Pro Git — Rerere | https://git-scm.com/book/en/v2/Git-Tools-Rerere |
| GitLab — GitLab Flow (canonical doc source) | https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/topics/gitlab_flow.md |
| GitLab — Merge trains | https://docs.gitlab.com/ci/pipelines/merge_trains/ (verified against canonical source: https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/ci/pipelines/merge_trains.md) |
| GitLab — Code Owners | https://docs.gitlab.com/user/project/codeowners/ |
| GitLab — CI job rules (`rules:changes`) | https://docs.gitlab.com/ci/jobs/job_rules/ (verified against canonical source: https://gitlab.com/gitlab-org/gitlab/-/blob/master/doc/ci/jobs/job_rules.md) |
| GitHub — GitHub flow | https://docs.github.com/en/get-started/using-github/github-flow |
| GitHub — About protected branches | https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches |
| GitHub — About code owners | https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners |
| GitHub — Managing a merge queue | https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue |
| GitHub — Actions workflow syntax (`paths`) | https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax |
| NVIDIA — DeepStream Reference Application (config groups) | https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html |
| NVIDIA — Gst-nvinfer | https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html |
| NVIDIA — Gst-nvdsanalytics | https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsanalytics.html |
| NVIDIA — Using a Custom Model with DeepStream | https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_using_custom_model.html |
| NVIDIA — nvdsinfer_custom_impl.h API reference | https://docs.nvidia.com/metropolis/deepstream/7.1/sdk-api/nvdsinfer__custom__impl_8h.html |
| NVIDIA — Gst-nvdsvideotemplate | https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsvideotemplate.html |
| NVIDIA — Implementing a Custom GStreamer Plugin (gst-dsexample) | https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_sample_custom_gstream.html |

### Sources consulted but not usable as citations
- Google, "Why Google Stores Billions of Lines of Code in a Single Repository" (CACM 2016) — abstract reachable at https://research.google/pubs/why-google-stores-billions-of-lines-of-code-in-a-single-repository/, but the full text returned HTTP 403 from ACM, so no claim from it is cited here. The only text retrievable was the abstract sentence: "Google's monolithic repository provides a common source of truth for tens of thousands of developers around the world."
- `docs.gitlab.com/topics/gitlab_flow/` intermittently 302-redirected to an auth endpoint during this research; GitLab Flow was therefore cited from GitLab's own canonical Markdown source in `gitlab-org/gitlab`.
