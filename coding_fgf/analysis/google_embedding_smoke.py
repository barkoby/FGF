from __future__ import annotations

import argparse
import json

from ..google_vertex import embed_texts, google_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", default="")
    parser.add_argument("--project-id", default="project_o")
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default="text-embedding-005")
    parser.add_argument("--text", default="minimal embedding smoke test")
    args = parser.parse_args()

    vectors = embed_texts(
        [args.text],
        args.model,
        google_config(args.project_id, args.location, args.credentials or None),
    )
    vector = vectors[0] if vectors else []
    print(json.dumps({"model": args.model, "dimension": len(vector), "first_values": vector[:3]}, sort_keys=True))


if __name__ == "__main__":
    main()
