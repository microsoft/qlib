import * as graphic from '../../util/graphic.js';
import SeriesData from '../../data/SeriesData.js';
import { AnimationOption, ZRColor, AnimationOptionMixin } from '../../util/types.js';
import { SymbolDrawSeriesScope } from './SymbolDraw.js';
import Model from '../../model/Model.js';
interface SymbolOpts {
    disableAnimation?: boolean;
    useNameLabel?: boolean;
    symbolInnerColor?: ZRColor;
}
declare class Symbol extends graphic.Group {
    private _symbolType;
    /**
     * Original scale
     */
    private _sizeX;
    private _sizeY;
    private _z2;
    constructor(data: SeriesData, idx: number, seriesScope?: SymbolDrawSeriesScope, opts?: SymbolOpts);
    _createSymbol(symbolType: string, data: SeriesData, idx: number, symbolSize: number[], keepAspect: boolean): void;
    /**
     * Stop animation
     * @param {boolean} toLastFrame
     */
    stopSymbolAnimation(toLastFrame: boolean): void;
    getSymbolType(): string;
    /**
     * FIXME:
     * Caution: This method breaks the encapsulation of this module,
     * but it indeed brings convenience. So do not use the method
     * unless you detailedly know all the implements of `Symbol`,
     * especially animation.
     *
     * Get symbol path element.
     */
    getSymbolPath(): import("../../util/symbol").ECSymbol;
    /**
     * Highlight symbol
     */
    highlight(): void;
    /**
     * Downplay symbol
     */
    downplay(): void;
    /**
     * @param {number} zlevel
     * @param {number} z
     */
    setZ(zlevel: number, z: number): void;
    setDraggable(draggable: boolean, hasCursorOption?: boolean): void;
    /**
     * Update symbol properties
     */
    updateData(data: SeriesData, idx: number, seriesScope?: SymbolDrawSeriesScope, opts?: SymbolOpts): void;
    _updateCommon(data: SeriesData, idx: number, symbolSize: number[], seriesScope?: SymbolDrawSeriesScope, opts?: SymbolOpts): void;
    setSymbolScale(scale: number): void;
    fadeOut(cb: () => void, seriesModel: Model<AnimationOptionMixin>, opt?: {
        fadeLabel: boolean;
        animation?: AnimationOption;
    }): void;
    static getSymbolSize(data: SeriesData, idx: number): [number, number];
}
export default Symbol;
