"""`hw <step>`: the pipeline's steps, each resumable. Run them in the order of the README."""

import argparse
from pathlib import Path

from . import config


def main() -> None:
    parser = argparse.ArgumentParser(prog="hw")
    parser.add_argument("--root", type=Path, default=Path("."), help="working directory for data")
    parser.add_argument("--clips", default="clips", help="name of the clip build to write or read")
    steps = parser.add_subparsers(dest="step", required=True)

    synth = steps.add_parser("synth", help="synthesize positives or confusables")
    synth.add_argument("kind", choices=["positive", "confusable"])
    synth.add_argument("--engine", choices=["piper", "pocket"], required=True)
    synth.add_argument("--count", type=int, required=True, help="clips per Piper voice / per Pocket language")
    synth.add_argument("--seed", type=int, default=0)
    synth.add_argument("--threads", type=int, default=4)
    synth.add_argument("--languages", nargs="+", help="Pocket only: these languages")
    synth.add_argument("--extra", choices=["lone", "round2"], help="another text list: the lone \"Hex\" or round-2 confusables")

    steps.add_parser("negatives", help="download and embed the long speech sources")
    steps.add_parser("noise", help="download room impulse responses and noises")

    build = steps.add_parser("clips", help="augment and embed the synthesized clips")
    build.add_argument("--variants", type=int, default=3, help="augmented copies per clip")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--lone-pause", action="store_true", help="the lone \"Hex\" fires only after a pause")

    train = steps.add_parser("train", help="train the classifier")
    train.add_argument("--run", default="run1")
    train.add_argument("--steps", type=int, default=20_000)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--phrases", choices=["all", "two-words"], default="all")

    evaluate = steps.add_parser("evaluate", help="recall and false accepts per hour on held-out data")
    evaluate.add_argument("--run", default="run1")
    evaluate.add_argument("--rules", action="store_true", help="measure the sensitivities of config.SENSITIVITY")

    export = steps.add_parser("export", help="write the ONNX model and its card")
    export.add_argument("--run", default="run1")

    args = parser.parse_args()
    paths = config.Paths(args.root.resolve(), args.clips)

    if args.step == "synth":
        from . import synth as s

        if args.engine == "piper":
            s.piper(paths, args.kind, args.count, seed=args.seed, extra=args.extra)
        else:
            s.pocket(
                paths,
                args.kind,
                args.count,
                languages=args.languages,
                seed=args.seed,
                threads=args.threads,
                extra=args.extra,
            )
    elif args.step == "negatives":
        from . import data, features

        embedder = features.Embedder(features.model_path(paths.models))
        for source in data.SOURCES:
            data.embed_source(paths, source, embedder)
    elif args.step == "noise":
        from . import data

        root = data.rirs_noises(paths)
        print(f"{len(data.rir_files(root))} impulse responses, {len(data.noise_files(root))} noises")
    elif args.step == "clips":
        from . import clips

        clips.build(paths, variants=args.variants, seed=args.seed, lone_pause=args.lone_pause)
    elif args.step == "train":
        from . import train as t

        t.train(paths, args.run, steps=args.steps, seed=args.seed, phrases=args.phrases)
    elif args.step == "evaluate":
        from . import evaluate as e

        if args.rules:
            e.evaluate_rules(paths, args.run)
        else:
            e.evaluate(paths, args.run)
    elif args.step == "export":
        from . import export as x

        x.export(paths, args.run)


if __name__ == "__main__":
    main()
