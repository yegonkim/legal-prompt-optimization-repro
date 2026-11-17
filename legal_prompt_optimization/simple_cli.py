"""Manifest-free experiment CLI; development never reads TEST."""
import argparse
import json
from pathlib import Path
from .simple_backend import SimpleBackend
from .simple_experiment import SearchConfig, TaskSpec, run_experiment, report_test


def read_rows(path, limit=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [dict(row, record_id=row.get('record_id', row.get('id'))) for row in rows[:limit]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['develop', 'report-test', 'preflight'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    backend = SimpleBackend(**config['backend'])
    if args.command == 'preflight':
        print(json.dumps(backend.preflight(), indent=2))
        return
    if args.data is None or args.out is None:
        parser.error('--data and --out are required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(args.out) + '.config.json').open('x') as stream:
        json.dump(config, stream, indent=2)
    if args.command == 'report-test':
        if args.selection is None:
            parser.error('--selection is required for TEST reporting')
        selection = json.loads(args.selection.read_text())
        task = TaskSpec(kind=selection['task']['kind'], labels=tuple(selection['task']['labels']))
        result = report_test(backend, read_rows(args.data / 'test.jsonl'), task, selection, args.out)
    else:
        splits = {name: read_rows(args.data / filename, limit) for name, filename, limit in
                  [('fit', 'train.jsonl', config.get('fit_size', 128)),
                   ('select', 'select.jsonl', config.get('select_size', 64)),
                   ('dev', 'dev.jsonl', None)]}
        first = splits['fit'][0]
        kind = first.get('task_type', 'classification')
        labels = tuple(dict.fromkeys(label for rows in splits.values() for row in rows for label in row.get('public_labels', [])))
        task = TaskSpec(kind=kind, labels=labels)
        result = run_experiment(backend, splits, task,
            config.get('ordinary_prompt', 'Solve the legal problem carefully and explain your reasoning.'),
            config.get('judge_prompt', 'Evaluate the four complete responses and determine the best answer.'),
            [SearchConfig(**item) for item in config['search']], args.out)
    print(json.dumps(result, indent=2))
    if result.get('failures'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
