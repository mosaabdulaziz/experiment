# Research — One repo or many, for several products on one shared ML/CV inference pipeline

Date: 2026-09-21
Scope: primary sources only — engineering blogs and documentation published by the company that runs the system, first-party product documentation, and first-party repository documentation. Secondary blog summaries, conference recaps and third-party "monorepo vs polyrepo" articles were deliberately excluded.

## Access notes (read this before trusting any quote below)

Some canonical sources are not machine-retrievable from this environment. They are named here rather than paraphrased from memory:

- **"Why Google Stores Billions of Lines of Code in a Single Repository", Potvin & Levenberg, Communications of the ACM 59(7):78–87, 2016.** The Google Research landing page is retrievable and gives the abstract and metadata; the page links to the ACM Digital Library for the full text and the full text is **not** served from Google's own page. <https://research.google/pubs/why-google-stores-billions-of-lines-of-code-in-a-single-repository/>
  - `cacm.acm.org` and `dl.acm.org` both returned **HTTP 403** to automated fetches on 2026-09-20 (<https://cacm.acm.org/research/why-google-stores-billions-of-lines-of-code-in-a-single-repository/>, <https://dl.acm.org/doi/10.1145/2854146>). CACM's own site may be readable in a browser; it was not readable here. **Therefore the only Google monorepo claims quoted below are from the abstract on research.google**, plus separately-accessible Google first-party documentation. Nothing from the body of the CACM article is paraphrased.
  - The companion first-party conference talk (Rachel Potvin, @Scale 2015) is published at <https://www.youtube.com/watch?v=W71BTkUbdqE>. No published transcript or slide deck was located, so it is cited as existing, not quoted.
- **Netflix Technology Blog** (`netflixtechblog.com`) is Medium-hosted and returned **HTTP 403** to automated fetches, including "Towards true continuous integration: distributed repositories and dependencies" (<https://netflixtechblog.com/towards-true-continuous-integration-distributed-repositories-and-dependencies-2a2e3108c051>). **No Netflix claim is made in this document.**
- **Airbnb Tech Blog** ("Migrating Airbnb's JVM Monorepo to Bazel", <https://medium.com/airbnb-engineering/migrating-airbnbs-jvm-monorepo-to-bazel-33f90eda51ec>) is Medium-hosted and returned **HTTP 403**. **No Airbnb claim is quoted below.**
- **Twitter/X engineering blog** ("Hello Pants build", 2014) returned **HTTP 403** on both `blog.twitter.com` and `blog.x.com` (<https://blog.x.com/engineering/en_us/a/2014/hello-pants-build>). A search-index snippet attributed the sentence *"Another solution is to have a single large, unified codebase. We've found that this promotes better engineering team cohesion and collaboration, which results in greater productivity and happiness"* to that post, but **the page itself could not be verified here, so this is flagged as unverified and not relied on.**

---

## 1. Monorepo practice at scale, stated by the companies themselves

### Google

The only directly-retrievable first-party text is the abstract:

> "Google's monolithic repository provides a common source of truth for tens of thousands of developers around the world. This article outlines the scale of Google's codebase, describes Google's custom-built monolithic source repository, and discusses the reasons behind choosing this model."
> — <https://research.google/pubs/why-google-stores-billions-of-lines-of-code-in-a-single-repository/>

Google does publish, first-party and freely, the code-review rules that describe the *review-scope cost* a monorepo imposes, because in a single repository a cross-cutting change is one changelist touching many owners' files:

> "Small, simple CLs are: Reviewed more quickly... Reviewed more thoroughly... Less likely to introduce bugs... Less wasted work if they are rejected."
> "100 lines is usually a reasonable size for a CL, and 1000 lines is usually too large, but it's up to the judgment of your reviewer."
> "The number of files that a change is spread across also affects its 'size.' A 200-line change in one file might be okay, but spread across 50 files it would usually be too large."
> "Reviewers have discretion to reject your change outright for the sole reason of it being too large."
> — Google Engineering Practices, <https://google.github.io/eng-practices/review/developer/small-cls.html>

This is the concrete, citable version of the "code review scope" cost: a monorepo makes an atomic cross-cutting change *possible*, and Google's own review policy then makes it *expensive* when it is wide.

### Meta

Meta states the monorepo decision explicitly, and states it as a rejection of splitting:

> "Breaking up the repository was also out of the question, as it would mean losing monorepo's benefits, such as simplified dependency management and the ability to make broad changes quickly."
> — "Sapling: Source control that's user-friendly and scalable", <https://engineering.fb.com/2022/11/15/open-source/sapling-source-control-scalable/>

The 2014 post makes the same argument, and adds the principle that tooling limits should not drive code layout:

> "there are a number of benefits to using a single repository. Even at our current scale, we often make large changes throughout our code base, and having a single repository is useful for continuous modernization."
> "Splitting it up would make large, atomic refactorings more difficult."
> "the idea that the scaling constraints of our source control system should dictate our code structure just doesn't sit well with us."
> — "Scaling Mercurial at Facebook", <https://engineering.fb.com/2014/01/07/core-infra/scaling-mercurial-at-facebook/>

The **cost** Meta reports is a purpose-built source-control stack. From the Sapling post: *"we've developed a virtual file system (not yet publicly available) that makes it look and act as if you have the entire repository"*, plus *"the scalable, distributed source control Rust service we use at Meta"*, and lazy fetching — *"cloning the repository downloads almost no history. Instead, as you use the repository we download just the commits, trees, and files you actually need."* (<https://engineering.fb.com/2022/11/15/open-source/sapling-source-control-scalable/>). Sapling's own docs enumerate the components that had to be built: *"New server infrastructure for push throughput and faster indexes (Mononoke, 2017)"*, *"Virtualized working copy for on-demand currently checked out file or tree fetching (EdenFS, 2018)"*, *"Faster commit graph algorithms (segmented changelog, 2020)"*, *"On-demand commit fetching (2021)"* — <https://sapling-scm.com/docs/scale/overview/>. Scale is given as *"tens of millions of files and commits, with tens of thousands of contributors"* (same page).

Meta's 2025 post on branching inside the monorepo is directly relevant to multi-product repos, because it states what a repo-wide branch costs when only one product needs to diverge:

> "full-repo branches are all-or-nothing. If you need to patch a legacy version, or maintain a custom variant for a particular project, you cannot create a branch for the part that you own. Branching forks everything."
> "full-repo merges create merge commits with multiple parents, making the commit graph _wide_ (high branching factor) and _non-linear_. In large monorepos, this creates performance problems for operations like sl log and sl blame."
> "having a single version of code remains the default assumption for the monorepo"
> — "Branching in a Sapling Monorepo", <https://engineering.fb.com/2025/10/16/developer-tools/branching-in-a-sapling-monorepo/>

That post names the circumstances under which Meta considers *directory* branching (a sub-tree diverging while the rest of the repo does not): expensive CI, experimental changes, and Git migrations (same URL).

Meta also publishes the clearest first-party account of what happens when several products each carry their own copy of a shared media/RTC core. It migrated *"over 50 use cases from a divergent WebRTC fork to a modular architecture built on top of the latest upstream version"*, and describes the divergence trap:

> "over time, as the upstream project evolves and your internal features accumulate, the resources needed to merge in external commits can become prohibitive."
> — "Escaping the Fork: How Meta Modernized WebRTC Across 50+ Use Cases", <https://engineering.fb.com/2026/04/09/developer-tools/escaping-the-fork-how-meta-modernized-webrtc-across-50-use-cases/>

The remedy Meta describes is *not* "one repo vs many repos" but a **shim layer between the application layer and WebRTC** that dispatches to legacy or latest implementation at runtime, letting many products sit on one shared core while migrating at different speeds (same URL). This is the single most transferable finding in this document for a shared CV pipeline with several products on it.

### Microsoft

Microsoft is the only one of these companies to publish the "how many repos" debate as a debate:

> "The first big debate was – how many repos do you have – one for the whole company at one extreme or one for each small component?"
> "Ever worked in a huge code base for 20 years? Ever tried to go back afterwards and decompose it into small repos? You can guess what we discovered. The code is very hard to decompose."
> "After much hand wringing we decided our strategy needed to be 'the right number of repos based on the character of the code'."
> "Sometimes, in big highly related code bases, it really is better to treat the codebase as a whole."
> — Brian Harry, "Scaling Git (and some back story)", <https://devblogs.microsoft.com/bharry/scaling-git-and-some-back-story/>

Scale and the cost of the choice:

> Windows is "approximately 3.5M files" and "a repo of about 300GB"; the team generates "8,421 pushes per day (on average)" with roughly 4,000 engineers. Previously, "in Source Depot, it was spread across 40+ depots and we had a tool to manage operations that spanned them."
> — "The largest Git repo on the planet", <https://devblogs.microsoft.com/bharry/the-largest-git-repo-on-the-planet/>

The tooling bill: Microsoft built GVFS/VFS for Git and then Scalar, which *"accelerates your Git workflow"* by *"setting recommended config values and running background maintenance"*, because *"Git is a distributed version control system, so by default each Git repository has a copy of all files in the entire history"* and *"time spent waiting for `git status` to report modified files or `git fetch` to get the latest data is time wasted"* — <https://devblogs.microsoft.com/devops/introducing-scalar/>. The same post cites the Windows OS repository as *"over three million"* tracked files and *"over 100 GB of compressed data"*.

**Note the Source Depot detail**: Microsoft's *prior* state was 40+ separate depots with a custom tool to span them. The monorepo replaced a cross-repo coordination tool — that is the trade being made, not an abstract preference.

### Uber

Uber's Go monorepo is the best-documented case of *many independently-deployed services from one repository*, which is the exact shape of a multi-product shared-pipeline repo:

> The Go monorepo launched in early 2018. It contains "over 70,000 files"; monthly commits grew "from less than 100 to nearly 10,000"; active developers grew "from under 10 to nearly 900"; and go.mod/go.sum reference "over 1,000" external modules.
> — "Building Uber's Go Monorepo with Bazel", <https://www.uber.com/us/en/blog/go-monorepo-bazel/>

The reported costs are all tooling: Make and `go build` *"no longer met our needs"* at that scale, so Uber adopted Bazel, contributed Gazelle features to cope with 1,000+ external modules, built a build-target-hashing tool to integrate with SubmitQueue (their merge queue that keeps master green), and upstreamed a Bazel feature to read build targets from a file (same URL).

Critically, Uber states that **release cadence is decoupled from repository layout**:

> Deployment is commit-scoped per service — "by scoping each service to the subset of commits that actually changed the code binary, it is also much easier for service owners to identify what exact changes their service is picking up with each deployment." Each pipeline stage "is run independently of any other. For each stage, the newest commit that has successfully completed the prior stage... is thus immediately advanced to the next stage." Weekly production deployments rose from 7,000 to 50,000 while commit volume only doubled.
> — "Continuous deployment for large monorepos", <https://www.uber.com/us/en/blog/continuous-deployment/>

Uber also publishes first-party material on the CI cost specifically — "How We Halved Go Monorepo CI Build Time" (<https://www.uber.com/us/en/blog/how-we-halved-go-monorepo-ci-build-time/>) and "Slashing CI Costs at Uber" (<https://www.uber.com/il/en/blog/slashing-ci-costs-at-uber/>) — which is the honest statement of the monorepo bill: CI on a monorepo is a standing engineering project, not a config file.

### Twitter/X and Airbnb

See Access notes. Both companies' relevant posts were unreachable (403). **No claims are made for either.**

---

## 2. The polyrepo / multi-repo counter-case, from first-party sources

### Amazon / AWS

AWS publishes the ownership argument directly:

> "no team should be big enough that it would take more than two pizzas to feed them"
> "single-threaded ownership over a specific product or service"
> "Two-pizza teams do not hand over something they've launched to another team to run."
> "smaller teams minimize lines of communication and decrease overhead of bureaucracy"
> "we fundamentally changed our technical architecture to what became known as a microservices architecture. We decoupled our monolithic architecture into a vast network of single, standalone services"
> — "Amazon's Two Pizza Teams", AWS Executive Insights, <https://aws.amazon.com/executive-insights/content/amazon-two-pizza-team/>

AWS Prescriptive Guidance states the pattern *and its stated disadvantages*, which is the balanced first-party statement this section needs:

> "the service per team pattern breaks them down into microservices that are managed by individual teams. Each team is responsible for a business capability and owns the capability's code base. The team independently develops, tests, deploys, or scales its services, and primarily interacts with other teams to negotiate APIs. We recommend that you assign each microservice to a single team."
> Advantages: "Teams act independently with minimal coordination." / "Code bases and microservices are not shared by multiple teams." / "Teams can quickly innovate and iterate on product features." / "Different teams can use different technologies, frameworks, or programming languages... These should be hidden behind a well-defined and stable API."
> Disadvantages: "It can be difficult to align teams to end-user functionality or business capabilities." / "Additional effort is required to deliver larger, coordinated application increments, especially if there are circular dependencies between teams."
> — AWS Prescriptive Guidance, "Service per team pattern", <https://docs.aws.amazon.com/prescriptive-guidance/latest/modernization-decomposing-monoliths/service-per-team.html>

Note what AWS does **not** claim: it does not claim independent repositories are cheaper. It claims independent *ownership* is cheaper, and says coordinated multi-service increments cost extra.

### Spotify

Spotify is the strongest first-party measurement of the polyrepo tax, because it puts a number on version drift of a shared core library:

> "Spotify currently uses a polyrepo layout with thousands of GitHub repos — although we have good reasons to believe that the same set of practices and tools outlined here will be applicable should we move to a monorepo in the future."
> "the number of distinct components we run in production has grown and is now on the order of thousands"
> — "Fleet Management at Spotify (Part 1)", <https://engineering.atspotify.com/2023/04/spotifys-shift-to-a-fleet-first-mindset-part-1>

> "Updating every one of these repos for every change or bugfix in the framework or core libraries would require a big effort, and, in the past, this could not be done quickly — as mentioned in the previous post, it would usually take 6+ months for each new release of the backend service framework to reach 70%+ adoption in our backend service fleet."
> — "Fleet Management at Spotify (Part 3): Fleet-wide Refactoring", <https://engineering.atspotify.com/2023/05/fleet-management-at-spotify-part-3-fleet-wide-refactoring>

Spotify's answer was not consolidation but automation: **Fleetshift** ("runs a Docker image against each repo"), an **Automerger** that merges a PR automatically when "the code change passes all tests and checks", plus Fleetsweep (pre-flight testing) and Firewatch (post-merge monitoring) — same URL. Spotify also notes the limit of that approach: *"a significant portion of our codebase resides in monorepos, where many teams are working in the same repo (such as our client/mobile codebases) — a workflow that is not adequately supported by Fleetshift today."*

So Spotify's own published position is **mixed-mode**: polyrepo for backend services, monorepos for the client/mobile codebases. That is a first-party example of the same company choosing differently for different product families.

### Netflix

See Access notes — unreachable, **no claim made**.

---

## 3. The middle ground: one authoritative source tree, versioned artifacts published outward

This is the pattern most product families actually run, and it has excellent first-party documentation.

### Kubernetes: `staging/` directories in one repo, published to read-only per-product repos by a bot

This is the strongest single primary source for "separate directories, separately consumable".

> "This directory is the staging area for packages that have been split to their own repository. The content here will be periodically published to respective top-level k8s.io repositories."
> "The code in the staging/ directory is authoritative, i.e. the only copy of the code. You can directly modify such code."
> "Kubernetes code uses the repositories in this directory via a Go workspace and module `replace` statements. For example, when Kubernetes code imports a package from the `k8s.io/client-go` repository, that import is resolved to `staging/src/k8s.io/client-go` relative to the project root"
> — kubernetes/kubernetes `staging/README.md`, <https://github.com/kubernetes/kubernetes/blob/master/staging/README.md> (raw: <https://raw.githubusercontent.com/kubernetes/kubernetes/master/staging/README.md>)

The README lists ~32 staged components (k8s.io/api, apimachinery, client-go, kubectl, kubelet, cri-api, component-base, metrics, …) — i.e. **dozens of separately-published artifacts living as sibling directories in one repository**. The creation procedure it documents is explicit that the published repo is a delivery endpoint, not a development location: *"Make sure that the `.github/PULL_REQUEST_TEMPLATE.md` and `CONTRIBUTING.md` files mention that PRs are not directly accepted to the repo"*, and *"The published repository **must** have an initial empty commit"* (same URL).

The published side confirms the contract:

> "This is an automatically published staged repository for Kubernetes."
> "This repository is still a mirror of k8s.io/kubernetes/staging/src/client-go, the code development is still done in the staging area."
> "This repository is read-only for importing, and not used for direct contributions."
> "Please send pull requests against the client packages in the Kubernetes main repository."
> — kubernetes/client-go README, <https://github.com/kubernetes/client-go/blob/master/README.md>

And the versioning contract that lets consumers pin: client-go carries its own `v0.x.y` tags aligned to Kubernetes releases, with a published compatibility matrix, and *"We will backport bugfixes--but not new features--into older versions of client-go"* (same URL).

The mechanism is `git filter-branch`-based one-way replication: the publishing bot *"pulls the latest k8s.io/kubernetes changes and runs `git filter-branch` to distill the commits that affect a staging repo"*, tagging each published commit with a `Kubernetes-sha:` reference — <https://github.com/kubernetes/publishing-bot>.

**Why this matters for the question**: Kubernetes gets monorepo properties for development (one atomic change across client-go + apimachinery + kubelet, one CI, one review) *and* polyrepo properties for consumption (each component is an independently pinnable, independently versioned module with its own compatibility guarantees). It did not have to choose.

### Bazel: modules, versions and pinning as the consumption contract

Bazel's own definitions make the middle-ground vocabulary precise:

> External dependency: "Source files (both text and binary) used in your build that are not from your workspace."
> Module: "A Bazel project that can have multiple versions, each of which can have dependencies on other modules."
> Repository: "A directory tree with a boundary marker file at its root, containing source files that can be used in a Bazel build."
> Workspace: "The environment shared by all Bazel commands run in the same main repository. It encompasses the main repo and the set of all defined external repos."
> Direct dependencies are declared in `MODULE.bazel`, e.g. `bazel_dep(name = "rules_cc", version = "0.1.1")`. External repos "are fetched on demand"; Vendor Mode lets you "pre-fetch the exact external dependencies you need to facilitate offline builds."
> — Bazel, "External dependencies overview", <https://bazel.build/external/overview>

Bzlmod resolves versions with **Minimal Version Selection**: *"MVS assumes that all new versions of a module are backwards compatible, and so picks the highest version specified by any dependent"*, giving a *"version selection process that is high-fidelity and reproducible"* — <https://bazel.build/external/module>.

Bazel's handling of **breaking changes in a shared module** is the most instructive part, because Bazel tried the machine-enforced approach and then withdrew it. It previously offered `compatibility_level`, which was to be incremented *"in the same commit that introduces a backwards incompatible ('breaking') change"*. Bazel's own FAQ now says:

> "You should stop using `compatibility_level`."
> "Increasing `compatibility_level` leads to version conflicts that are difficult for end users to resolve."
> "Starting with Bazel 8.6.0 and 9.1.0, both `compatibility_level` and `max_compatibility_level` are no-ops."
> "Module maintainers introducing major breaking changes should ensure that build failures provide clear error messages and actionable migration paths."
> — Bazel external dependencies FAQ, <https://bazel.build/external/faq>

That is a first-party admission that a hard version-incompatibility signal across independently-pinned consumers pushes the cost onto those consumers, and that the maintainer's job is migration paths rather than enforcement.

### ROS: packages as the unit, repositories as a packaging convenience

ROS 2's developer guide declines to mandate one package per repo, and ties grouping to *release* coupling:

> "Each package should be in a subfolder which has the same name as the package. If a repository contains only a single package it can optionally be in the root of the repository."
> "Major version increments (i.e. breaking changes) should not be made within a released ROS distribution."
> "Patch (interface-preserving) and minor (non-breaking) version increments do not break compatibility, so these sorts of changes *are* allowed within a release."
> "We will version all libraries within a package together. This means that libraries inherit their version from the package." — with the stated reason: "This keeps library and package versions from diverging and shares reasoning with the policy of releasing packages which share a repository together."
> — ROS 2 Developer Guide, <https://docs.ros.org/en/rolling/The-ROS2-Project/Contributing/Developer-Guide.html> (source: <https://github.com/ros2/ros2_documentation/blob/rolling/source/The-ROS2-Project/Contributing/Developer-Guide.rst>)
> (Note: the rendered docs.ros.org page returned an "Access Denied"/Anubis challenge to automated fetching; the quotes above are from the ReStructuredText source in the ros2_documentation repository, which is the same first-party content.)

The load-bearing sentence is the last one: **things that share a repository get released together.** That is ROS's stated criterion, and it is the cleanest one-line rule any of these sources give.

---

## 4. Decision criteria stated by first-party sources

Each row is a criterion a primary source actually states, with the direction it points.

| Criterion | What the source says | Points toward | Source |
|---|---|---|---|
| **Are changes usually cross-cutting?** | "we often make large changes throughout our code base, and having a single repository is useful for continuous modernization"; "Splitting it up would make large, atomic refactorings more difficult." | one repo | [Meta 2014](https://engineering.fb.com/2014/01/07/core-infra/scaling-mercurial-at-facebook/) |
| **Cost of propagating a shared-core change** | "it would usually take 6+ months for each new release of the backend service framework to reach 70%+ adoption in our backend service fleet" | one repo, or heavy automation | [Spotify Part 3](https://engineering.atspotify.com/2023/05/fleet-management-at-spotify-part-3-fleet-wide-refactoring) |
| **Cost of a divergent copy of the shared core** | "over time, as the upstream project evolves and your internal features accumulate, the resources needed to merge in external commits can become prohibitive" | one shared core + runtime shim, not forks | [Meta WebRTC](https://engineering.fb.com/2026/04/09/developer-tools/escaping-the-fork-how-meta-modernized-webrtc-across-50-use-cases/) |
| **Release cadence coupling** | "releasing packages which share a repository together"; "Major version increments... should not be made within a released ROS distribution" | split only what must release on its own clock | [ROS 2 Developer Guide](https://docs.ros.org/en/rolling/The-ROS2-Project/Contributing/Developer-Guide.html) |
| **Release cadence is *not* automatically coupled to layout** | per-service commit-scoped deploys from one monorepo; each pipeline stage "is run independently of any other"; 7,000 → 50,000 weekly deploys | one repo is compatible with independent cadence *if* you build the deploy system | [Uber CD](https://www.uber.com/us/en/blog/continuous-deployment/) |
| **Ownership boundaries** | "single-threaded ownership over a specific product or service"; "Two-pizza teams do not hand over something they've launched to another team to run" | per-product ownership (not necessarily per-product repo) | [AWS two-pizza](https://aws.amazon.com/executive-insights/content/amazon-two-pizza-team/) |
| **Coordination overhead vs coordinated increments** | Advantages: "Teams act independently with minimal coordination." Disadvantages: "Additional effort is required to deliver larger, coordinated application increments, especially if there are circular dependencies between teams." | depends on whether increments are coordinated | [AWS service-per-team](https://docs.aws.amazon.com/prescriptive-guidance/latest/modernization-decomposing-monoliths/service-per-team.html) |
| **Code review scope** | "100 lines is usually a reasonable size for a CL, and 1000 lines is usually too large"; a 200-line change "spread across 50 files... would usually be too large"; reviewers may reject "for the sole reason of it being too large" | wide changes need splitting regardless of layout | [Google eng-practices](https://google.github.io/eng-practices/review/developer/small-cls.html) |
| **CI / build cost** | Make and `go build` "no longer met our needs"; required Bazel + Gazelle + merge-queue integration + target hashing; separate published work on halving CI time and cutting CI cost | one repo has a standing CI engineering bill | [Uber Bazel](https://www.uber.com/us/en/blog/go-monorepo-bazel/), [Uber CI](https://www.uber.com/us/en/blog/how-we-halved-go-monorepo-ci-build-time/) |
| **SCM tooling bill at scale** | GVFS/VFS for Git, then Scalar; Mononoke + EdenFS + segmented changelog | only pay this if you are at that scale | [Microsoft Scalar](https://devblogs.microsoft.com/devops/introducing-scalar/), [Sapling scale](https://sapling-scm.com/docs/scale/overview/) |
| **Is the code actually decomposable?** | "Ever tried to go back afterwards and decompose it into small repos?... The code is very hard to decompose"; the strategy became "the right number of repos based on the character of the code" | measure coupling, then decide | [Brian Harry](https://devblogs.microsoft.com/bharry/scaling-git-and-some-back-story/) |
| **Does one product need to diverge while others don't?** | "full-repo branches are all-or-nothing... you cannot create a branch for the part that you own. Branching forks everything"; single version of code "remains the default assumption" | directory-level divergence, not repo split | [Meta branching](https://engineering.fb.com/2025/10/16/developer-tools/branching-in-a-sapling-monorepo/) |
| **Breaking change in the shared core** | Bazel withdrew its machine-enforced break signal: "You should stop using `compatibility_level`"; "Increasing `compatibility_level` leads to version conflicts that are difficult for end users to resolve"; maintainers should instead "ensure that build failures provide clear error messages and actionable migration paths" | hard breaks across pinned consumers are expensive; prefer migration paths | [Bazel FAQ](https://bazel.build/external/faq) |
| **Consumers need to pin** | published components carry their own `v0.x.y` tags + compatibility matrix; "We will backport bugfixes--but not new features--into older versions" | publish versioned artifacts from one tree | [client-go](https://github.com/kubernetes/client-go/blob/master/README.md) |

---

## 5. Computer vision specifically: what NVIDIA's DeepStream documentation says

NVIDIA's own documentation does **not** describe a per-application repository model. It describes **one SDK source tree, many applications as sibling directories, and per-application behaviour expressed as configuration plus pluggable libraries.**

### 5.1 One reference application, driven by config, not by a fork

> "The DeepStream reference application is a GStreamer based solution and consists of set of GStreamer plugins encapsulating low-level APIs to form a complete graph."
> "The NVIDIA DeepStream SDK reference application uses one of the sample configuration files from the `samples/configs/deepstream-app` directory in the DeepStream package to: Enable or disable components, Change the properties or behavior of components, Customize other application configuration settings."
> — "DeepStream Reference Application - deepstream-app", <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html>

The configuration is a set of named groups — application, tiled-display, source, streammux, preprocess, primary-gie, secondary-gie, tracker, message-converter, osd, sink, tests, nvds-analytics (same URL). Per-plugin configuration specs (Gst-nvdspreprocess, Gst-nvinfer, Gst-nvtracker, Gst-nvdewarper, Gst-nvmsgconv, Gst-nvmsgbroker, Gst-nvdsanalytics) are documented in the Plugin Guide (<https://docs.nvidia.com/metropolis/deepstream/dev-guide/>).

**Reading**: NVIDIA's unit of product differentiation is the config file plus the model, not a repository.

### 5.2 Custom per-product logic goes in a plugin/library, enabled by a config group

> "a mechanism to add third party or custom algorithms within the reference application by modifying the example plugin (`gst-dsexample`). The sources for the plugin are in `sources/gst-plugins/gst-dsexample` directory in the SDK."
> Enablement is by adding a config section — `[ds-example]` with `enable=1`, `processing-width`, `processing-height`, `full-frame`, `blur-objects`, `unique-id` — to an existing configuration file.
> Custom logic is injected by replacing the calls `DsExampleCtxInit`, `DsExampleCtxDeinit`, `DsExampleProcess`, `blur_objects` "with corresponding functions of any other custom library".
> — "Implementing a Custom GStreamer Plugin with OpenCV Integration Example", <https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_sample_custom_gstream.html>

So NVIDIA's documented extension seam is: **shared binary pipeline + your custom library behind a stable plugin interface + a config group that switches it on.** That is the same shape as Meta's WebRTC shim, arrived at independently.

### 5.3 Service Maker states reuse across applications explicitly

> "Deepstream Service Maker provides a high-level abstraction layer on top of the NVIDIA Deepstream SDK, designed to simplify the development of real-time multimedia processing and AI inference applications."
> — <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_intro.html>

> "A plugin encapsulates a specialized factory, enabling the creation of user-implemented custom objects through a standardized factory API."
> Plugins are defined with `DS_CUSTOM_PLUGIN_DEFINE()` / `DS_CUSTOM_FACTORY_DEFINE()`, built as a shared library with CMake, and discovered at runtime from default install paths or a path given by `NVDS_MODULE_PATH`.
> The documented purpose: "seamless reuse of user-defined custom objects (e.g., object counter, latency measurement, etc) across multiple applications."
> — "What is a Deepstream Service Maker Plugin", <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_plugin.html>

This is NVIDIA stating the middle-ground pattern in its own words: **the shared unit is a runtime-loadable plugin, and applications are thin consumers of it.** Not a repo per app.

### 5.4 How NVIDIA lays out its own app collections

- **C/C++ sample apps** ship inside the SDK tree at `/opt/nvidia/deepstream/deepstream/sources`, with `apps/sample_apps` holding `deepstream-app`, `deepstream-test1` … `deepstream-test5`, plus specialised apps (dewarper, optical flow, image-meta, analytics, 3D, nmos, server, ucx-test); `apps/apps-common` holds shared helpers; `gst-plugins/` holds the plugin sources (nvinfer, nvinferserver, nvdspreprocess, nvtracker, nvdsosd, nvmsgconv/nvmsgbroker, dsexample); `libs/` holds NvDsInfer, NvDsInferServer, message converters, protocol adapters (Kafka, AMQP, Azure), REST server and DS3D components. — "C/C++ Sample Apps Source Details", <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_C_Sample_Apps.html>
- **Service Maker C++ sample apps** live at `/opt/nvidia/deepstream/deepstream/service-maker/sources/apps/cpp/`. — <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_cpp.html>
- **Python apps are the one place NVIDIA does split a repository**, and the split is by *language/binding*, not by product: they live in `NVIDIA-AI-IOT/deepstream_python_apps`, with every app as a sibling directory under `apps/` (`deepstream-test1`…`test4`, `deepstream-imagedata-multistream`, `deepstream-opticalflow`, `deepstream-segmentation`, `runtime_source_add_delete`, `deepstream-preprocess-test`, `deepstream-rtsp-in-rtsp-out`, `deepstream-demux-multi-in-multi-out`, `deepstream-segmask`, …) and the pybind11 bindings in a sibling `bindings/` directory. The docs note *"The SDK MetaData library is developed in C/C++. Python bindings provide access to the MetaData from Python applications"* and that this path is deprecated in favour of `pyservicemaker`. — <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html>, <https://github.com/NVIDIA-AI-IOT/deepstream_python_apps>
- **Reference/vertical applications** (anomaly detection, runtime source add/delete, MaskTracker/SAM2, single- and multi-view 3D tracking, parallel multi-model inferencing, bodypose-3D, IPC buffer sharing, dynamic sources with a single decoder, custom video tiling, VLLM plugin, plus a `legacy_apps/` folder) are all sibling directories inside **one** repository, each with its own README — *"For further details, please see each project's README."* — <https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps>

**Bottom line for §5**: across three separate NVIDIA-owned collections, the layout is *one repository, one app per directory, shared plugins/libs at the top*. The only repository boundaries NVIDIA draws are (a) SDK vs. community samples and (b) C/C++ vs. Python bindings. **No NVIDIA documentation located here suggests a repository per application.**

---

## 6. Decision criteria applied to a shared DeepStream pipeline with 3–5 products (attendance, fire/smoke, barber shop, productivity)

Stated as criteria-with-source. No scale numbers are invented; where a source's number is not transferable, that is said.

1. **Scale does not justify a split here.** Every monorepo-cost citation above is about repositories at a scale this project is nowhere near — Windows at ~3.5M files / ~300GB ([Brian Harry](https://devblogs.microsoft.com/bharry/the-largest-git-repo-on-the-planet/)), Meta at "tens of millions of files and commits" ([Sapling](https://sapling-scm.com/docs/scale/overview/)), Uber's Go monorepo at "over 70,000 files" ([Uber](https://www.uber.com/us/en/blog/go-monorepo-bazel/)). The tooling bills those companies report (GVFS/Scalar, Mononoke/EdenFS, Bazel+Gazelle+merge queue) are the *costs of a monorepo*, and none of them are triggered at 3–5 products. The monorepo costs documented by primary sources therefore do not apply; the polyrepo costs (next point) do.

2. **The shared core is the pipeline, and the products differ mostly by model + config — which is the case NVIDIA documents as config-driven, not repo-driven.** DeepStream's documented mechanism for a different use case on the same graph is a different configuration file selecting sources, streammux, primary/secondary GIE, tracker, analytics and sinks (<https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html>), with per-product logic behind a plugin interface enabled by its own config group (<https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_sample_custom_gstream.html>). On NVIDIA's own model, "attendance", "fire/smoke", "barber shop" and "productivity" are **configurations and plugins over one pipeline**, not four applications.

3. **The polyrepo failure mode is exactly the one this shape invites.** Spotify's measured figure — *"it would usually take 6+ months for each new release of the backend service framework to reach 70%+ adoption"* ([Spotify Part 3](https://engineering.atspotify.com/2023/05/fleet-management-at-spotify-part-3-fleet-wide-refactoring)) — is the cost of shipping a shared-core change into independently-versioned consumers. That number is Spotify's, at thousands of repos; it is **not** a prediction for 3–5 products. What transfers is the direction, not the magnitude: the lag is non-zero and grows with the number of pinned consumers.

4. **Divergent copies of a shared CV/media core are the specific thing a first-party source warns against.** Meta's WebRTC post is the closest analogue in the sources: 50+ use cases on one media core, and *"over time, as the upstream project evolves and your internal features accumulate, the resources needed to merge in external commits can become prohibitive"* (<https://engineering.fb.com/2026/04/09/developer-tools/escaping-the-fork-how-meta-modernized-webrtc-across-50-use-cases/>). This repository already vendors three perception/models/DeepStream-Yolo variants (`perception/models/DeepStream-Yolo`, `perception/models/DeepStream-Yolo-Face`, `perception/models/DeepStream-Yolo-Fire`), which is the fork-divergence shape Meta describes. Meta's documented remedy is a **shim/dispatch layer over a single shared core**, not one repo per consumer.

5. **Release cadence is the ROS criterion, and it is the one that could justify a split.** ROS states the rule plainly: things that "share a repository" are released "together", and *"Major version increments (i.e. breaking changes) should not be made within a released ROS distribution"* (<https://docs.ros.org/en/rolling/The-ROS2-Project/Contributing/Developer-Guide.html>). So: if attendance must ship to a customer on a frozen version while fire/smoke ships weekly, that is the *only* criterion in this document that argues for separating them — and Meta's counter-argument is that a repo-wide branch is the wrong tool for it, because *"Branching forks everything"* and *"you cannot create a branch for the part that you own"* (<https://engineering.fb.com/2025/10/16/developer-tools/branching-in-a-sapling-monorepo/>).

6. **Independent deployment does not require independent repositories.** Uber deploys per service out of one monorepo by scoping each service to "the subset of commits that actually changed the code binary", with each pipeline stage run "independently of any other" (<https://www.uber.com/us/en/blog/continuous-deployment/>). If independent per-product release is the requirement, the primary-source answer is a per-product deploy pipeline, not a per-product repository.

7. **If per-product pinning is genuinely needed, the documented pattern is publish-from-one-tree.** Kubernetes keeps ~32 components as sibling directories where *"The code in the staging/ directory is authoritative, i.e. the only copy of the code"*, and a bot publishes each to a read-only repo that is *"read-only for importing, and not used for direct contributions"*, each carrying its own version tags and compatibility matrix (<https://github.com/kubernetes/kubernetes/blob/master/staging/README.md>, <https://github.com/kubernetes/client-go/blob/master/README.md>). This gives atomic cross-cutting edits *and* per-consumer pinning, and is the shape to copy if a product ever needs to hold back on an older core.

8. **A breaking change in the shared core is costly precisely in proportion to how many consumers pin it.** Bazel offered a machine-enforced incompatibility signal (`compatibility_level`) and then withdrew it, stating *"You should stop using `compatibility_level`"* because *"Increasing `compatibility_level` leads to version conflicts that are difficult for end users to resolve"*, and that maintainers should instead *"ensure that build failures provide clear error messages and actionable migration paths"* (<https://bazel.build/external/faq>). Applied here: as long as all products build from one tree, a breaking change to the pipeline is one change with one set of call sites. The moment products pin versions of the core, that same change becomes a migration campaign — which is the Spotify cost in point 3.

9. **Ownership, not repository count, is what AWS actually argues for.** AWS's stated benefit is *"single-threaded ownership over a specific product or service"* and teams that *"act independently with minimal coordination"*, with the stated cost that *"Additional effort is required to deliver larger, coordinated application increments"* (<https://aws.amazon.com/executive-insights/content/amazon-two-pizza-team/>, <https://docs.aws.amazon.com/prescriptive-guidance/latest/modernization-decomposing-monoliths/service-per-team.html>). With a single owner across all these products, the autonomy benefit AWS describes is not available to be bought, while the coordination cost it describes is still payable.

10. **Review scope stays a real cost either way.** Google's published rule — a 200-line change across 50 files "would usually be too large", and reviewers may reject a change "for the sole reason of it being too large" (<https://google.github.io/eng-practices/review/developer/small-cls.html>) — applies to a wide change regardless of how the code is laid out. Splitting repositories does not make a wide change smaller; it makes it several changes that must land in order.

**What the sources jointly point to for this case:** one repository, one directory per product, with the pipeline, plugins and shared libraries at the top — mirroring NVIDIA's own three collections (<https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_C_Sample_Apps.html>, <https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps>, <https://github.com/NVIDIA-AI-IOT/deepstream_python_apps>) — with per-product behaviour carried in configuration and runtime-loaded plugins (<https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_plugin.html>), and the Kubernetes staging/publishing pattern held in reserve for the day one product must pin an older core (<https://github.com/kubernetes/kubernetes/blob/master/staging/README.md>). The one fact that would overturn this is divergent, externally-imposed release freezes per product, which is the criterion ROS states (<https://docs.ros.org/en/rolling/The-ROS2-Project/Contributing/Developer-Guide.html>).

---

## Source list

**Retrieved and quoted**
- Google Research, abstract only — <https://research.google/pubs/why-google-stores-billions-of-lines-of-code-in-a-single-repository/>
- Google Engineering Practices, small CLs — <https://google.github.io/eng-practices/review/developer/small-cls.html>
- Meta, Scaling Mercurial at Facebook — <https://engineering.fb.com/2014/01/07/core-infra/scaling-mercurial-at-facebook/>
- Meta, Sapling — <https://engineering.fb.com/2022/11/15/open-source/sapling-source-control-scalable/>
- Meta, Branching in a Sapling Monorepo — <https://engineering.fb.com/2025/10/16/developer-tools/branching-in-a-sapling-monorepo/>
- Meta, Escaping the Fork (WebRTC, 50+ use cases) — <https://engineering.fb.com/2026/04/09/developer-tools/escaping-the-fork-how-meta-modernized-webrtc-across-50-use-cases/>
- Sapling docs, scale overview — <https://sapling-scm.com/docs/scale/overview/>
- Microsoft, Scaling Git (and some back story) — <https://devblogs.microsoft.com/bharry/scaling-git-and-some-back-story/>
- Microsoft, The largest Git repo on the planet — <https://devblogs.microsoft.com/bharry/the-largest-git-repo-on-the-planet/>
- Microsoft, Introducing Scalar — <https://devblogs.microsoft.com/devops/introducing-scalar/>
- Uber, Building Uber's Go Monorepo with Bazel — <https://www.uber.com/us/en/blog/go-monorepo-bazel/>
- Uber, Continuous deployment for large monorepos — <https://www.uber.com/us/en/blog/continuous-deployment/>
- Uber, How We Halved Go Monorepo CI Build Time — <https://www.uber.com/us/en/blog/how-we-halved-go-monorepo-ci-build-time/>
- Spotify, Fleet Management Part 1 — <https://engineering.atspotify.com/2023/04/spotifys-shift-to-a-fleet-first-mindset-part-1>
- Spotify, Fleet Management Part 3 — <https://engineering.atspotify.com/2023/05/fleet-management-at-spotify-part-3-fleet-wide-refactoring>
- AWS, Amazon's Two Pizza Teams — <https://aws.amazon.com/executive-insights/content/amazon-two-pizza-team/>
- AWS Prescriptive Guidance, Service per team pattern — <https://docs.aws.amazon.com/prescriptive-guidance/latest/modernization-decomposing-monoliths/service-per-team.html>
- Kubernetes, staging README — <https://github.com/kubernetes/kubernetes/blob/master/staging/README.md>
- Kubernetes, client-go README — <https://github.com/kubernetes/client-go/blob/master/README.md>
- Kubernetes, publishing-bot — <https://github.com/kubernetes/publishing-bot>
- Bazel, External dependencies overview — <https://bazel.build/external/overview>
- Bazel, Bazel modules / MVS — <https://bazel.build/external/module>
- Bazel, external dependencies FAQ (compatibility_level deprecation) — <https://bazel.build/external/faq>
- ROS 2 Developer Guide — <https://docs.ros.org/en/rolling/The-ROS2-Project/Contributing/Developer-Guide.html> (fetched from the .rst source in <https://github.com/ros2/ros2_documentation>)
- NVIDIA, DeepStream reference application — <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html>
- NVIDIA, Custom GStreamer plugin (gst-dsexample) — <https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_sample_custom_gstream.html>
- NVIDIA, C/C++ Sample Apps Source Details — <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_C_Sample_Apps.html>
- NVIDIA, Python Sample Apps and Bindings — <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html>
- NVIDIA, Service Maker intro — <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_intro.html>
- NVIDIA, Service Maker plugin — <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_plugin.html>
- NVIDIA, Service Maker for C/C++ — <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_cpp.html>
- NVIDIA-AI-IOT, deepstream_reference_apps — <https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps>
- NVIDIA-AI-IOT, deepstream_python_apps — <https://github.com/NVIDIA-AI-IOT/deepstream_python_apps>

**Named but NOT retrievable from this environment (no claims made from them)**
- CACM / ACM DL full text of Potvin & Levenberg 2016 — 403 — <https://cacm.acm.org/research/why-google-stores-billions-of-lines-of-code-in-a-single-repository/>, <https://dl.acm.org/doi/10.1145/2854146>
- Rachel Potvin @Scale 2015 talk — video only, no published transcript/slides found — <https://www.youtube.com/watch?v=W71BTkUbdqE>
- Netflix Technology Blog, "Towards true continuous integration" — 403 (Medium) — <https://netflixtechblog.com/towards-true-continuous-integration-distributed-repositories-and-dependencies-2a2e3108c051>
- Airbnb Tech Blog, "Migrating Airbnb's JVM Monorepo to Bazel" — 403 (Medium) — <https://medium.com/airbnb-engineering/migrating-airbnbs-jvm-monorepo-to-bazel-33f90eda51ec>
- Twitter/X Engineering, "Hello Pants build" (2014) and "The release of Pants 1.0" (2016) — 403 — <https://blog.x.com/engineering/en_us/a/2014/hello-pants-build>, <https://blog.x.com/engineering/en_us/a/2016/the-release-of-pants-10>
