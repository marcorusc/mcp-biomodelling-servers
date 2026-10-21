"""Transactional portable configuration bundles; no executable packaging."""
import hashlib
import json
import os
import tempfile
import uuid
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

from mcp_biomodelling_servers.artifact_manager import (
    get_artifact_dir,
    safe_artifact_path,
)
from mcp_biomodelling_servers.handoff import load_handoff_manifest

from .configuration import revision


def export_bundle(session, root, working_directory=None):
    config = deepcopy(session.config)
    source_revision = revision(config)
    xml = ET.fromstring(config.generate_xml())
    lineage = []
    for context in session.maboss_contexts.values():
        if context.local_manifest_path:
            verified = load_handoff_manifest(context.local_manifest_path, verify_artifacts=True)
            lineage.append(verified.model_dump(mode="json"))
    base = Path(working_directory) if working_directory else None
    if base is not None and (not base.is_absolute() or not base.is_dir()):
        raise ValueError("working_directory must name the source executable's absolute working directory")
    artifact_dir = get_artifact_dir(root, session.session_id)
    bundle_id = "bundle_" + uuid.uuid4().hex
    destination = safe_artifact_path(artifact_dir, bundle_id + ".zip")
    with tempfile.TemporaryDirectory(prefix=".physicell-bundle-", dir=artifact_dir) as temporary:
        staging = Path(temporary)
        (staging / "config").mkdir()
        entries = []
        generated_rule_path = None
        if config.cell_rules.get_rules():
            generated_rule_path = staging / "config/generated_rules.csv"
            config.cell_rules.generate_csv(str(generated_rule_path))
            ruleset_name = session.generated_ruleset_name or "mcp_generated_" + uuid.uuid4().hex
            config.cell_rules.add_ruleset(ruleset_name, folder=str(generated_rule_path.parent),
                                         filename=generated_rule_path.name, enabled=True)
            xml = ET.fromstring(config.generate_xml())

        handled = {id(child) for element in xml.iter() if element.get("enabled", "").lower() == "false"
                   for child in element.iter()}

        def include(element, relative, folder=None):
            text = (element.text or "").strip()
            source = Path(text)
            if folder is not None:
                source = Path((folder.text or "").strip()) / source
            if not source.is_absolute():
                if base is None:
                    raise ValueError(f"Relative input {source} needs explicit working_directory; XML location is not assumed")
                source = base / source
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Missing or symlinked required input: {source}")
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            registered = session.initial_cells_artifact
            if registered and str(source.resolve()) == str(Path(registered["path"]).resolve()):
                if digest != registered["sha256"] or len(data) != registered["size_bytes"]:
                    raise ValueError("Registered initial-cell CSV integrity mismatch")
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            entries.append({"path": relative, "source": "stored_cell_rules" if source == generated_rule_path else str(source), "sha256": digest, "size_bytes": len(data)})
            if folder is None:
                element.text = relative
            else:
                folder.text = str(Path(relative).parent)
                element.text = Path(relative).name
                handled.add(id(folder))
            handled.add(id(element))

        for index, cell in enumerate(xml.findall("./cell_definitions/cell_definition")):
            intracellular = cell.find("./phenotype/intracellular")
            if intracellular is None:
                continue
            if intracellular.get("type") != "maboss":
                raise ValueError(f"Unsupported intracellular dependency type for {cell.get('name')}")
            for tag, extension in (("bnd_filename", "bnd"), ("cfg_filename", "cfg")):
                element = intracellular.find(tag)
                if element is None or not element.text:
                    raise ValueError(f"Missing {tag} for {cell.get('name')}")
                include(element, f"config/boolean_models/cell_{index}.{extension}")
        cells = xml.find("./initial_conditions/cell_positions")
        if cells is not None:
            if cells.get("enabled", "false").lower() == "true":
                if cells.get("type", "csv") != "csv":
                    raise ValueError("Only CSV cell positions are supported in portable bundles")
                filename, folder = cells.find("filename"), cells.find("folder")
                if filename is None or folder is None:
                    raise ValueError("Initial cells need folder and filename")
                include(filename, "config/cells.csv", folder)
            else:
                handled.update(id(e) for e in cells.iter())
        rulesets = xml.findall("./cell_rules/rulesets/ruleset")
        for index, ruleset in enumerate(rulesets):
            if ruleset.get("enabled", "true").lower() == "false":
                handled.update(id(e) for e in ruleset.iter())
                continue
            filename, folder = ruleset.find("filename"), ruleset.find("folder")
            if filename is None or folder is None or ruleset.get("format", "csv") != "csv":
                raise ValueError("Unsupported cell rules input")
            include(filename, f"config/rules_{index}.csv", folder)
        # Refuse unimplemented input-reference features rather than silently omit them.
        for element in xml.iter():
            if (element.tag == "filename" or element.tag.endswith("_filename")) and id(element) not in handled:
                raise ValueError(f"Unsupported external input reference: {element.tag}={element.text}")
        if generated_rule_path is not None:
            generated_rule_path.unlink()
        xml_path = staging / "config/PhysiCell_settings.xml"
        ET.indent(xml)
        ET.ElementTree(xml).write(xml_path, encoding="utf-8", xml_declaration=True)
        data = xml_path.read_bytes()
        entries.append({"path": "config/PhysiCell_settings.xml", "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
        manifest = {"schema_version": "1.0", "bundle_id": bundle_id, "session_id": session.session_id,
                    "model_revision": source_revision, "status": "configuration_complete", "scientific_approval": "not_granted",
                    "working_directory": "bundle_root", "upstream_lineage": lineage, "files": entries,
                    "substrate_origins": session.substrate_origins,
                    "requirements": ["Existing compatible PhysiCell/PhysiBoSS executable and template", "Scientific review remains required"]}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "README.md").write_text(
            "# PhysiCell configuration bundle\n\nExtract and use the extracted bundle root as the working directory.\n"
            "Pass config/PhysiCell_settings.xml to an existing compatible executable.\n"
            "All packaged input paths are relative to that working directory, not the XML folder.\n"
            "No executable is included and no simulation or scientific validation was performed.\n"
            "The executable/template must support the configured cell types, behaviors, CSV format and PhysiBoSS version.\n"
            "User parameters, cell counts and seeds are interpreted by that template; CSV can be additive.\n")
        archive = staging / "bundle.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for file in staging.rglob("*"):
                if file.is_file() and file != archive:
                    out.write(file, file.relative_to(staging))
        os.link(archive, destination)  # unique and complete publication, no overwrite
    return {"session_id": session.session_id, "model_revision": source_revision, "bundle_id": bundle_id,
            "bundle": str(destination), "manifest": "manifest.json (inside bundle)",
            "status": "configuration_complete", "requirements": manifest["requirements"]}
