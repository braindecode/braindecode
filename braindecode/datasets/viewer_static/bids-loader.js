/* ============================================================
   bids-loader.js — parse BIDS electrodes.tsv + coordsystem.json.

   Originally hosted a full montage-builder pipeline (sphere fit,
   axis normalisation, region inference). That code was retired
   when bids-recording.js absorbed the registry/montage path; only
   the two parsers below remain in use (see bids-recording.js:735,
   bids-recording.js:739). Earlier history lives in git.
   ============================================================ */
(function () {
  'use strict';

  const api = {};

  // ---- TSV parsing --------------------------------------------
  api.parseElectrodesTSV = function (text) {
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length < 2) throw new Error('electrodes.tsv has no rows');

    const headers = lines[0].split('\t').map(h => h.trim().toLowerCase());
    const col = (name) => headers.indexOf(name);
    const iName = col('name'), iX = col('x'), iY = col('y'), iZ = col('z');
    if (iName < 0 || iX < 0 || iY < 0 || iZ < 0) {
      throw new Error('electrodes.tsv is missing one of: name, x, y, z');
    }
    const iType = col('type'), iMat = col('material');

    // Optional BIDS columns we preserve if present: `coordinate_system` and
    // `group` drive multi-frame EMG panelling; `hemisphere` helps iEEG.
    const iCoordSys = headers.indexOf('coordinate_system');
    const iGroup = headers.indexOf('group');
    const iHemi = headers.indexOf('hemisphere');

    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const c = lines[i].split('\t');
      const name = (c[iName] || '').trim();
      const x = parseFloat(c[iX]);
      const y = parseFloat(c[iY]);
      const z = parseFloat(c[iZ]);
      // BIDS uses "n/a" for missing; parseFloat → NaN → skip.
      if (!name || !isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
      const row = {
        name, x, y, z,
        type: iType >= 0 ? (c[iType] || '').trim() : '',
        material: iMat >= 0 ? (c[iMat] || '').trim() : '',
      };
      if (iCoordSys >= 0 && c[iCoordSys] && c[iCoordSys].trim() && c[iCoordSys].trim().toLowerCase() !== 'n/a') {
        row.coordinate_system = c[iCoordSys].trim();
      }
      if (iGroup >= 0 && c[iGroup] && c[iGroup].trim() && c[iGroup].trim().toLowerCase() !== 'n/a') {
        row.group = c[iGroup].trim();
      }
      if (iHemi >= 0 && c[iHemi] && c[iHemi].trim() && c[iHemi].trim().toLowerCase() !== 'n/a') {
        row.hemisphere = c[iHemi].trim();
      }
      rows.push(row);
    }
    if (rows.length < 4) throw new Error('Need at least 4 electrodes with finite x,y,z');
    return rows;
  };

  // ---- coordsystem.json ---------------------------------------
  api.parseCoordsystem = function (jsonOrText) {
    const obj = typeof jsonOrText === 'string' ? JSON.parse(jsonOrText) : jsonOrText;
    // BIDS prefixes coordinate keys by datatype: EEGCoordinateSystem,
    // iEEGCoordinateSystem, MEGCoordinateSystem, EMGCoordinateSystem (BEP-030),
    // NIRSCoordinateSystem. Pick whichever prefix has a match.
    const prefixes = ['EEG', 'iEEG', 'MEG', 'EMG', 'NIRS'];
    const prefix = prefixes.find(
      p => obj[p + 'CoordinateSystem'] || obj[p + 'CoordinateUnits']
    ) || 'EEG';
    return {
      space: obj[prefix + 'CoordinateSystem'] || 'Other',
      units: (obj[prefix + 'CoordinateUnits'] || 'm').toLowerCase(),
      landmarks: obj.AnatomicalLandmarkCoordinates || null,
    };
  };

  if (typeof window !== 'undefined') window.BIDSLoader = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof globalThis !== 'undefined') globalThis.BIDSLoader = api;
})();
