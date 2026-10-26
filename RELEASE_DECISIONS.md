# Release decisions requiring owner confirmation

Original review recorded 2026-10-24. On 2026-10-26, the account owner requested publication of four selected repositories; this repository was selected. The earlier blanket instruction to keep all repositories private is superseded for the selected four. Merge authorization is being resolved separately.

The questions below remain a record of unverified licensing, data-access and scientific details. Publication authorization from the account owner must not be represented as independent proof of institutional permission, a dataset license or full scientific reproduction.
Passing execution tests does not answer the following questions. Potential contributors named below are parties to verify, not a legal determination of ownership.

## IGBT-R — Code rights and license

Confirm author/institution and any experimental/simulation partner permission for the Transformer, physical feature equations and degradation-level definitions. Identify any inherited model code requiring attribution.

Required response: identify the rights holder and approver, the permitted code/legacy-file scope, the written approval or applicable license, and the exact project-wide license authorized. State whether publication of embedded constants and source provenance is permitted. No project-wide license will be inferred from a paper or a nearby license file.

## IGBT-D — Data and disclosure

For the four healthy/6.66/13.33/20 spreadsheets, identify the source owner, units and meaning of these percentages, data terms and whether exact column names/physical parameters may be disclosed.

Required response: dataset name/version, rights holder, access URL or contact process, permitted uses and redistribution terms. Choose **code plus synthetic data only**, **code plus an approved request-access route**, or **code plus specifically licensed data**. Approve disclosure of schemas/identifiers separately. Until answered, no research data or trained weights are included.

## IGBT-M — Scientific scope

Confirm class definitions and the final paper split/model configuration. The runner uses compact chronological classification and reports accuracy; class-index MSE does not measure ordinal severity quality. Confirm that SHAP, full baselines and the reserved validation block are outside this release, or supply a validated protocol for them.

Required response: approve the documented limited scope, or supply the authoritative equations/script/experimental protocol needed for a change. Smoke tests are not evidence that the scientific choice is correct.

## IGBT-H — Original Git metadata

The original commit retains a local machine email/hostname. New audit commits use the GitHub noreply address. Confirm either (a) retaining the original metadata, or (b) preparing a separately reviewed sanitized public history with required authorship/attribution preserved. No existing branch/history will be force-rewritten as part of this review.

## IGBT-A — Archive disclosure

Choose whether the eventual public release should include the selected `legacy/` files, a reviewed/redacted subset, or only the maintained runner plus provenance/attribution. Review historical paths, filenames, names and scientific constants before deciding. Original research folders remain unchanged.

Record decisions here with approver, date, scope and evidence reference. Do not paste confidential agreements or original datasets into this repository. An answer to these questions is not an instruction to change repository visibility or merge this PR.
