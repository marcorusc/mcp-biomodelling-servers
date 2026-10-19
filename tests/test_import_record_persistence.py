"""Durable import provenance and failure-safe record publication."""

import pytest

from mcp_biomodelling_servers import import_lineage as lineage


def test_import_records_survive_source_edits_and_later_exports(tmp_path):
    source = tmp_path / 'input.sif'
    source.write_text('A\t1\tB\n')
    first = lineage.capture_import(source, server='NeKo', session_id='child')
    event = lineage.write_source_import(tmp_path, first)
    original = event.read_bytes()
    lineage.write_source_export(source, server='NeKo', session_id='child', imports=[first])
    source.write_text('A\t-1\tB\n')
    second = lineage.capture_import(source, server='NeKo', session_id='child')
    next_event = lineage.write_source_import(tmp_path, second)
    lineage.write_source_export(source, server='NeKo', session_id='child', imports=[second])
    assert event != next_event
    assert event.read_bytes() == original
    assert lineage.SourceImportRecord.model_validate_json(original).imported == first
    assert lineage.SourceImportRecord.model_validate_json(next_event.read_bytes()).imported == second
    assert first.source.sha256 != second.source.sha256


def test_publication_failure_never_replaces_previous_record(tmp_path, monkeypatch):
    source = tmp_path / 'model.bnet'
    source.write_text('A, A\n')
    sidecar = lineage.write_source_export(source, server='NeKo', session_id='parent', imports=[])
    original = sidecar.read_bytes()
    with pytest.raises(FileExistsError):
        lineage.write_source_export(source, server='NeKo', session_id='parent', imports=[], overwrite=False)
    assert sidecar.read_bytes() == original
    imported = lineage.capture_import(source, server='NeKo', session_id='child')

    def fail_link(*args):
        raise OSError('publication failure')

    monkeypatch.setattr(lineage.os, 'link', fail_link)
    with pytest.raises(OSError, match='publication failure'):
        lineage.write_source_import(tmp_path, imported)
    assert not list(tmp_path.glob('source-import.*.json'))
    assert not list(tmp_path.glob('.source-record-*'))
    assert sidecar.read_bytes() == original


def test_sidecar_rejects_mismatched_import_ownership(tmp_path):
    source = tmp_path / 'model.bnet'
    source.write_text('A, A\n')
    imported = lineage.capture_import(source, server='NeKo', session_id='one')
    with pytest.raises(ValueError, match='ownership'):
        lineage.write_source_export(source, server='NeKo', session_id='two', imports=[imported])
    assert not (tmp_path / 'model.bnet.lineage.json').exists()


def test_sidecar_rejects_mismatched_ancestor_ownership(tmp_path):
    source = tmp_path / 'model.bnet'
    source.write_text('A, A\n')
    parent = lineage.write_source_export(source, server='NeKo', session_id='parent', imports=[])
    imported = lineage.capture_import(source, server='MaBoSS', session_id='child', parent_manifest=str(parent))
    data = imported.model_dump(mode='json')
    data['derived_from']['session_id'] = 'unrelated'
    with pytest.raises(ValueError, match='ownership'):
        lineage.SourceImport.model_validate(data)
