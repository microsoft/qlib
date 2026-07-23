import * as zrUtil from 'zrender/lib/core/util.js';
import { AllPropTypes, Dictionary } from 'zrender/lib/core/types.js';
import { ColorString, BuiltinVisualProperty, VisualOptionPiecewise, VisualOptionUnit, ParsedValue } from '../util/types.js';
declare type RawValue = ParsedValue;
declare type VisualValue = AllPropTypes<VisualOptionUnit>;
declare type NormalizedValue = number;
declare type MappingMethod = 'linear' | 'piecewise' | 'category' | 'fixed';
interface Normalizer {
    (this: VisualMapping, value?: RawValue): NormalizedValue;
}
interface ColorMapper {
    (this: VisualMapping, value: RawValue | NormalizedValue, isNormalized?: boolean, out?: number[]): ColorString | number[];
}
interface DoMap {
    (this: VisualMapping, normalzied?: NormalizedValue, value?: RawValue): VisualValue;
}
interface VisualValueGetter {
    (key: string): VisualValue;
}
interface VisualValueSetter {
    (key: string, value: VisualValue): void;
}
interface VisualHandler {
    applyVisual(this: VisualMapping, value: RawValue, getter: VisualValueGetter, setter: VisualValueSetter): void;
    _normalizedToVisual: {
        linear(this: VisualMapping, normalized: NormalizedValue): VisualValue;
        category(this: VisualMapping, normalized: NormalizedValue): VisualValue;
        piecewise(this: VisualMapping, normalzied: NormalizedValue, value: RawValue): VisualValue;
        fixed(this: VisualMapping): VisualValue;
    };
    /**
     * Get color mapping for the outside usage.
     * Currently only used in `color` visual.
     *
     * The last parameter out is cached color array.
     */
    getColorMapper?: (this: VisualMapping) => ColorMapper;
}
interface VisualMappingPiece {
    index?: number;
    value?: number | string;
    interval?: [number, number];
    close?: [0 | 1, 0 | 1];
    text?: string;
    visual?: VisualOptionPiecewise;
}
export interface VisualMappingOption {
    type?: BuiltinVisualProperty;
    mappingMethod?: MappingMethod;
    /**
     * required when mappingMethod is 'linear'
     */
    dataExtent?: [number, number];
    /**
     *  required when mappingMethod is 'piecewise'.
     *  Visual for only each piece can be specified
     * [
     *   {value: someValue},
     *   {interval: [min1, max1], visual: {...}},
     *   {interval: [min2, max2]}
     *  ],.
     */
    pieceList?: VisualMappingPiece[];
    /**
     * required when mappingMethod is 'category'. If no option.categories, categories is set as [0, 1, 2, ...].
     */
    categories?: (string | number)[];
    /**
     * Whether loop mapping when mappingMethod is 'category'.
     * @default false
     */
    loop?: boolean;
    /**
     * Visual data
     * when mappingMethod is 'category', visual data can be array or object
     * (like: {cate1: '#222', none: '#fff'})
     * or primary types (which represents default category visual), otherwise visual
     * can be array or primary (which will be normalized to array).
     */
    visual?: VisualValue[] | Dictionary<VisualValue> | VisualValue;
}
interface VisualMappingInnerPiece extends VisualMappingPiece {
    originIndex: number;
}
interface VisualMappingInnerOption extends VisualMappingOption {
    hasSpecialVisual: boolean;
    pieceList: VisualMappingInnerPiece[];
    /**
     * Map to get category index
     */
    categoryMap: Dictionary<number>;
    /**
     * Cached parsed rgba array from string to avoid parse every time.
     */
    parsedVisual: number[][];
    visual?: VisualValue[] | Dictionary<VisualValue>;
}
declare class VisualMapping {
    option: VisualMappingInnerOption;
    type: BuiltinVisualProperty;
    mappingMethod: MappingMethod;
    applyVisual: VisualHandler['applyVisual'];
    getColorMapper: VisualHandler['getColorMapper'];
    _normalizeData: Normalizer;
    _normalizedToVisual: DoMap;
    constructor(option: VisualMappingOption);
    mapValueToVisual(value: RawValue): VisualValue;
    getNormalizer(): zrUtil.Bind1<Normalizer, this>;
    static visualHandlers: {
        [key in BuiltinVisualProperty]: VisualHandler;
    };
    /**
     * List available visual types.
     *
     * @public
     * @return {Array.<string>}
     */
    static listVisualTypes(): ("symbol" | "color" | "opacity" | "decal" | "symbolSize" | "liftZ" | "colorAlpha" | "colorLightness" | "colorSaturation" | "colorHue")[];
    /**
     * @public
     */
    static isValidType(visualType: string): boolean;
    /**
     * Convenient method.
     * Visual can be Object or Array or primary type.
     */
    static eachVisual<Ctx, T>(visual: T | T[] | Dictionary<T>, callback: (visual: T, key?: string | number) => void, context?: Ctx): void;
    static mapVisual<Ctx, T>(visual: T, callback: (visual: T, key?: string | number) => T, context?: Ctx): T;
    static mapVisual<Ctx, T>(visual: T[], callback: (visual: T, key?: string | number) => T[], context?: Ctx): T[];
    static mapVisual<Ctx, T>(visual: Dictionary<T>, callback: (visual: T, key?: string | number) => Dictionary<T>, context?: Ctx): Dictionary<T>;
    /**
     * Retrieve visual properties from given object.
     */
    static retrieveVisuals(obj: Dictionary<any>): VisualOptionPiecewise;
    /**
     * Give order to visual types, considering colorSaturation, colorAlpha depends on color.
     *
     * @public
     * @param {(Object|Array)} visualTypes If Object, like: {color: ..., colorSaturation: ...}
     *                                     IF Array, like: ['color', 'symbol', 'colorSaturation']
     * @return {Array.<string>} Sorted visual types.
     */
    static prepareVisualTypes(visualTypes: {
        [key in BuiltinVisualProperty]?: any;
    } | BuiltinVisualProperty[]): (keyof VisualOptionUnit)[];
    /**
     * 'color', 'colorSaturation', 'colorAlpha', ... are depends on 'color'.
     * Other visuals are only depends on themself.
     */
    static dependsOn(visualType1: BuiltinVisualProperty, visualType2: BuiltinVisualProperty): boolean;
    /**
     * @param value
     * @param pieceList [{value: ..., interval: [min, max]}, ...]
     *                         Always from small to big.
     * @param findClosestWhenOutside Default to be false
     * @return index
     */
    static findPieceIndex(value: number, pieceList: VisualMappingPiece[], findClosestWhenOutside?: boolean): number;
}
export default VisualMapping;
