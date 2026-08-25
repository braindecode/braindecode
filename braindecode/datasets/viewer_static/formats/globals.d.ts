// Ambient declarations for the format-reader globals attached to
// globalThis by formats/_*.js helpers. These files are loaded as
// classic scripts in the browser and expose objects under fixed
// names; tsc --checkJs needs to know about them.
//
// All types are intentionally `any` — these are runtime-checked
// helper bundles and we don't want JSDoc maintenance overhead here.
// The goal of tsc on formats/ is to catch the public-API entry-point
// drift, not to type the internal helpers.

declare const HttpRange: any;
declare const ChannelBuffers: any;
declare const ChannelLabels: any;
declare const ChannelDecode: any;
declare const BIDSRecording: any;
declare const MatV5: any;
declare const Mat73: any;
declare const SidecarChecks: any;
declare const StreamingUtils: any;
declare const CTFRes4: any;
declare const CTFMarker: any;
declare const FiffDir: any;
declare const KitReader: any;
// Lane H: BIDS-allowed formats. See formats/<name>.js for support tier
// (full / metadata-only / stub).
declare const NwbReader: any;
declare const MefReader: any;
declare const MefSegment: any;
declare const BtiReader: any;
declare const BtiConfig: any;
declare const ItabReader: any;
declare const KrissReader: any;

// Some call sites reach through globalThis explicitly.
declare global {
  // eslint-disable-next-line no-var
  var HttpRange: any;
  // eslint-disable-next-line no-var
  var ChannelBuffers: any;
  // eslint-disable-next-line no-var
  var ChannelLabels: any;
  // eslint-disable-next-line no-var
  var ChannelDecode: any;
  // eslint-disable-next-line no-var
  var BIDSRecording: any;
  // eslint-disable-next-line no-var
  var MatV5: any;
  // eslint-disable-next-line no-var
  var Mat73: any;
  // eslint-disable-next-line no-var
  var SidecarChecks: any;
  // eslint-disable-next-line no-var
  var StreamingUtils: any;
  // eslint-disable-next-line no-var
  var CTFRes4: any;
  // eslint-disable-next-line no-var
  var CTFMarker: any;
  // eslint-disable-next-line no-var
  var FiffDir: any;
  // eslint-disable-next-line no-var
  var KitReader: any;
  // Lane H: BIDS-allowed formats.
  // eslint-disable-next-line no-var
  var NwbReader: any;
  // eslint-disable-next-line no-var
  var MefReader: any;
  // eslint-disable-next-line no-var
  var MefSegment: any;
  // eslint-disable-next-line no-var
  var BtiReader: any;
  // eslint-disable-next-line no-var
  var BtiConfig: any;
  // eslint-disable-next-line no-var
  var ItabReader: any;
  // eslint-disable-next-line no-var
  var KrissReader: any;
}

export {};
