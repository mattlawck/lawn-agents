# 2026-09-15 — The guardrail had a hole, and the bug had four layers

Ten days of work since the acceptance run. Two new ADRs on the
guardrails, a working task loop, and one debugging session that went
four layers deep and ended somewhere I did not expect.

Writing it down while the wrong turns are still fresh, because the wrong
turns are the interesting part.

## The guardrail I was proud of had a hole in it

ADR 0003 is the spine of this project: never guess on chemicals. Three
layers — prompt, schema, tests — and I have described the schema layer
as "the real enforcer" more than once, including in `architecture.md`.

The schema layer is one validator:

```python
@model_validator(mode="after")
def _requires_citation_for_chemicals(self) -> Self:
    if isinstance(self.category, ChemicalCategory) and not self.citations:
        raise ValueError(...)
```

It counts citations. It has never once looked at what they point at.

Two probes made that concrete. The first asked about doveweed and got
back a recommendation describing Celsius's active ingredients as
"thiencarbazone-methyl, **foramsulfuron**, dicamba."

`foramsulfuron` appears in zero of 311 corpus chunks. It is in no source
I own. And it arrived wrapped in a real, correctly-attributed citation
to the actual Celsius label — which states, in text I can quote back,
that the actives are thiencarbazone-methyl and iodosulfuron-methyl-sodium.

The second asked about goosegrass and got Celsius recommended, cited to
a general Clemson weed factsheet. The Celsius label in my corpus never
mentions goosegrass. Not once in thirty chunks.

Neither failure escalated. `is_weak` scored retrieval as strong, so no
lexical check ran, no gate fired, no research subagent woke up. The
system was confident, cited, and wrong.

The common root took me embarrassingly long to say out loud: **presence
of a citation is not support for a claim**, and nothing in the codebase
distinguished the two.

ADR 0010 adds the check that does — deterministic, no LLM, because a
guardrail built to catch a model's mistakes should not itself be a model
that can make them. It verifies the cited source was actually in
`<sources>`, that the quoted snippet appears in it, and that any
chemical the action names shows up in a cited passage.

It has since caught a real one in production. A first draft named
chemistry absent from its citations, the re-prompt landed on a refusal,
and no invented rate reached me. That is the whole job.

What it does *not* catch is the goosegrass case, and the ADR says so at
length. If a cited passage mentions both a weed and a chemistry in
unrelated contexts, pairing them still passes. That is a retrieval
problem wearing a grounding costume — which brings me to the part of
this week I will remember.

## Four layers, and three of them were symptoms

I had nutsedge to spray and both products in the shed. I asked the
system. It refused.

**Layer one: the labels weren't in the corpus.** Obviously. Ingested
both. Asked again. Still refused — correctly, since a citation needs a
source and the source had only just arrived.

**Layer two: the brand bridge never reached retrieval.** ADR 0007 taught
the *synthesizer* that "Sedge Ender" means sulfentrazone. Two weeks ago
I taught the *relevance check* the same thing. Retrieval was never
wired. So a question naming a product had never once searched for that
product's chemistry — the exact asymmetry ADR 0008 had already fixed for
weeds, sitting unnoticed on the other axis. Wired it. Still refused.

**Layer three: dense retrieval cannot find a product by name.** This is
where I stopped guessing and measured. The query

> "Sedge Ender sulfentrazone application rate per 1000 sq ft Zoysia matrella"

returned **zero Sedge Ender chunks** in its top eight. What it returned
was the rate tables of Fusilade II, Recognition and Celsius — three
other products. BGE-small reads "application rate per 1000 sq ft" as the
dominant signal, every label matches it, and a brand name is a rare
token carrying almost no semantic weight. It gets washed out.

That is not a phrasing problem. I tried three phrasings and reverted all
three rather than keep machinery that did not work. It is what dense
retrieval is bad at, and the textbook answer is exact-term matching.

So I added BM25. LanceDB has native full-text search, so no new
dependency. Searching `"Sedge Ender"` put the Sedge Ender label at rank
0, score 10.85 — precisely what embeddings could not do.

Still refused.

**Layer four: the rate tables do not contain the product's name.**

Of the twelve chunks in the Bonide Sedge Ender label, exactly **one**
contained the words "Sedge Ender" — and it was neither of the two
holding the application-rate tables. Those chunks open "SPECIFIC USE
INSTRUCTIONS IN TURFGRASS" and "Grass Type / Warm Season Grasses."

EPA labels refer to their subject as *"this product"* throughout. Of
course they do. The product's name is on the front panel; the document
never needs to repeat it.

No retrieval algorithm finds text that is not there. I had spent a day
tuning retrieval for a problem that was not in retrieval.

The fix is embarrassingly small: prepend the source title to every
chunk, so each one is self-identifying. The question answered on the
next run.

## The same change, wrong and then right

Mid-way through layer three I added "reserve a slot for each bridge
query's best hit" — RRF rewards consensus across queries, which is
wrong for a question naming two products, because a label chunk scores
on the single query naming its product and loses to chunks scoring
moderately on four.

It did not work. The reserved slots filled with the *wrong* products'
rate tables, because under vector-only retrieval each brand query's top
hit was the wrong document. So I reverted it, on the principle that
complexity which does not pay is debt.

Two hours later, after BM25 landed, I put the identical change back —
and it worked, because now each brand query's top hits were reliably the
right product.

Same diff. Opposite verdict. The only thing that changed was what sat
underneath it. I have written that into the ADR because it is the most
transferable thing I learned this week: "does this change work" is not a
property of the change.

## What the system actually said

The payoff, for a question I needed answered for real:

> **Spot spray actively growing nutsedge with Sedgehammer
> (halosulfuron-methyl).** Mix one water-soluble bag into 1 gallon of
> water, add 2 teaspoons of non-ionic surfactant. Covers ~1,000 sq ft.
>
> *Gated by:* nutsedge actively growing at 3–8 leaves, no rain expected
> for 24 hours, avoid windy days, 65–85°F. **Consider D2 drought —
> stress reduces control and increases turf damage.**

Cited to EPA label pages 3 and 4, NCSU, and Clemson. And then, without
being asked:

> "While Sedge Ender (sulfentrazone) is also effective… the provided
> sources do not contain specific application rates for residential spot
> spraying. Please consult your product's label."

A complete answer for the product it could ground, and transparency
about the one it could not. That is better than the refusal I was
trying to eliminate, and better than the confident answer I was afraid
of.

## The rest of the week, briefly

**The audit found things that had never worked.** The `climate:`
thresholds — green-up, dormancy, pre-emergent triggers — had sat in
`config.yaml` since Phase 1 and were read by no code, while both prompts
hardcoded the numbers and claimed they came from config. `tenacity` was
a declared dependency imported nowhere. `http.retries` was configured
and unwired. `top_k_vector` and `top_k_bm25` described a hybrid
retriever that did not exist — and then, two weeks later, did.

**The soil-temperature API was not flaky, it was impossible.** The
station lookup pulls 1.63 MB and takes 17–26 seconds against a 15-second
timeout. It fails more often than it succeeds, and it was being called
every run to recompute a haversine whose answer never changes. Pinning
the station took it to 0.66 s.

**A security finding I did not expect.** The research subagent validates
a URL's host against the allowlist, then fetches with redirects
followed, and never re-checked where it landed. An allowlisted host
redirecting elsewhere would have had its content ingested under the
*original* allowlisted `source_id` — inheriting `tier=extension` from
ADR 0009 and satisfying ADR 0010's grounding. Off-allowlist text
laundered into an authoritative citation.

**The task loop works.** Todoist is the state layer, tasks carry
`[item-id/year]` markers so de-duplication is exact, and the weekly
watchdog is silent unless a critical window is approaching or something
has slipped. It uses no LLM at all — deciding whether soil temperature
crossed a threshold is arithmetic — so it cannot hallucinate a
recommendation at 7am while I am asleep.

The first generated set put five tasks on one day and scheduled
pre-emergent five weeks early, because I had used the *window* start as
the due date when the window only means "start watching." Both were my
bugs and both came back as user feedback within a day, which is the
argument for using the thing you build.

## What the blog should say

1. **"Cited" and "supported" are different words.** The guardrail I
   trusted checked the first and I had been reading it as the second,
   in my own architecture doc, for months.
2. **Four layers, three symptoms.** Every fix was locally correct and
   the bug stayed. The actual cause was that EPA labels call themselves
   "this product" — a fact about documents, not about retrieval.
3. **A change can be wrong and then right without changing.** Reserved
   slots failed, got reverted, and succeeded unaltered two hours later
   on a different substrate.
4. **Config that isn't read is a lie with a version history.** The
   thresholds had been decorative for months while the prompt cited
   them as authority.
