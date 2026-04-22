import Path from '../graphic/Path';
import TSpan from '../graphic/TSpan';
export declare function normalizeLineDash(lineType: any, lineWidth?: number): number[] | false;
export declare function getLineDash(el: Path | TSpan): [number[] | false, number];
