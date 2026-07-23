import * as graphic from '../../util/graphic.js';
import SeriesData from '../../data/SeriesData.js';
import { StageHandlerProgressParams, LabelOption, SymbolOptionMixin, ItemStyleOption, ZRColor, AnimationOptionMixin, ZRStyleProps, StatesOptionMixin, BlurScope, DisplayState, DefaultEmphasisFocus } from '../../util/types.js';
import { CoordinateSystemClipArea } from '../../coord/CoordinateSystem.js';
import Model from '../../model/Model.js';
import Element from 'zrender/lib/Element.js';
import SeriesModel from '../../model/Series.js';
interface UpdateOpt {
    isIgnore?(idx: number): boolean;
    clipShape?: CoordinateSystemClipArea;
    getSymbolPoint?(idx: number): number[];
    disableAnimation?: boolean;
}
interface SymbolLike extends graphic.Group {
    updateData(data: SeriesData, idx: number, scope?: SymbolDrawSeriesScope, opt?: UpdateOpt): void;
    fadeOut?(cb: () => void, seriesModel: SeriesModel): void;
}
interface SymbolLikeCtor {
    new (data: SeriesData, idx: number, scope?: SymbolDrawSeriesScope, opt?: UpdateOpt): SymbolLike;
}
interface RippleEffectOption {
    period?: number;
    /**
     * Scale of ripple
     */
    scale?: number;
    brushType?: 'fill' | 'stroke';
    color?: ZRColor;
    /**
     * ripple number
     */
    number?: number;
}
interface SymbolDrawStateOption {
    itemStyle?: ItemStyleOption;
    label?: LabelOption;
}
export interface SymbolDrawItemModelOption extends SymbolOptionMixin<object>, StatesOptionMixin<SymbolDrawStateOption, {
    emphasis?: {
        focus?: DefaultEmphasisFocus;
        scale?: boolean | number;
    };
}>, SymbolDrawStateOption {
    cursor?: string;
    rippleEffect?: RippleEffectOption;
}
export interface SymbolDrawSeriesScope {
    emphasisItemStyle?: ZRStyleProps;
    blurItemStyle?: ZRStyleProps;
    selectItemStyle?: ZRStyleProps;
    focus?: DefaultEmphasisFocus;
    blurScope?: BlurScope;
    emphasisDisabled?: boolean;
    labelStatesModels: Record<DisplayState, Model<LabelOption>>;
    itemModel?: Model<SymbolDrawItemModelOption>;
    hoverScale?: boolean | number;
    cursorStyle?: string;
    fadeIn?: boolean;
}
export declare type ListForSymbolDraw = SeriesData<Model<SymbolDrawItemModelOption & AnimationOptionMixin>>;
declare class SymbolDraw {
    group: graphic.Group;
    private _data;
    private _SymbolCtor;
    private _seriesScope;
    private _getSymbolPoint;
    private _progressiveEls;
    constructor(SymbolCtor?: SymbolLikeCtor);
    /**
     * Update symbols draw by new data
     */
    updateData(data: ListForSymbolDraw, opt?: UpdateOpt): void;
    updateLayout(): void;
    incrementalPrepareUpdate(data: ListForSymbolDraw): void;
    /**
     * Update symbols draw by new data
     */
    incrementalUpdate(taskParams: StageHandlerProgressParams, data: ListForSymbolDraw, opt?: UpdateOpt): void;
    eachRendered(cb: (el: Element) => boolean | void): void;
    remove(enableAnimation?: boolean): void;
}
export default SymbolDraw;
