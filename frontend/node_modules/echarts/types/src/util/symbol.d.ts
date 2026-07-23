import * as graphic from './graphic.js';
import { Dictionary } from 'zrender/lib/core/types.js';
import { SymbolOptionMixin, ZRColor } from './types.js';
export declare type ECSymbol = graphic.Path & {
    __isEmptyBrush?: boolean;
    setColor: (color: ZRColor, innerColor?: ZRColor) => void;
    getColor: () => ZRColor;
};
export declare const symbolBuildProxies: Dictionary<ECSymbol>;
/**
 * Create a symbol element with given symbol configuration: shape, x, y, width, height, color
 */
export declare function createSymbol(symbolType: string, x: number, y: number, w: number, h: number, color?: ZRColor, keepAspect?: boolean): ECSymbol;
export declare function normalizeSymbolSize(symbolSize: number | number[]): [number, number];
export declare function normalizeSymbolOffset(symbolOffset: SymbolOptionMixin['symbolOffset'], symbolSize: number[]): [number, number];
