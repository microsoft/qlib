import { Group } from '../../util/graphic.js';
import SeriesData from '../../data/SeriesData.js';
import type { ZRColor } from '../../util/types.js';
interface RippleEffectCfg {
    showEffectOn?: 'emphasis' | 'render';
    rippleScale?: number;
    brushType?: 'fill' | 'stroke';
    period?: number;
    effectOffset?: number;
    z?: number;
    zlevel?: number;
    symbolType?: string;
    color?: ZRColor;
    rippleEffectColor?: ZRColor;
    rippleNumber?: number;
}
declare class EffectSymbol extends Group {
    private _effectCfg;
    constructor(data: SeriesData, idx: number);
    stopEffectAnimation(): void;
    startEffectAnimation(effectCfg: RippleEffectCfg): void;
    /**
     * Update effect symbol
     */
    updateEffectAnimation(effectCfg: RippleEffectCfg): void;
    /**
     * Highlight symbol
     */
    highlight(): void;
    /**
     * Downplay symbol
     */
    downplay(): void;
    getSymbolType(): string;
    /**
     * Update symbol properties
     */
    updateData(data: SeriesData, idx: number): void;
    fadeOut(cb: () => void): void;
}
export default EffectSymbol;
