import { ComponentOption, BoxLayoutOptionMixin, ZRTextAlign, ZRTextVerticalAlign, ZRColor, BorderOptionMixin, LabelOption } from '../../util/types.js';
import { EChartsExtensionInstallRegisters } from '../../extension.js';
export interface TitleOption extends ComponentOption, BoxLayoutOptionMixin, BorderOptionMixin {
    mainType?: 'title';
    show?: boolean;
    text?: string;
    /**
     * Link to url
     */
    link?: string;
    target?: 'self' | 'blank';
    subtext?: string;
    sublink?: string;
    subtarget?: 'self' | 'blank';
    textAlign?: ZRTextAlign;
    textVerticalAlign?: ZRTextVerticalAlign;
    /**
     * @deprecated Use textVerticalAlign instead
     */
    textBaseline?: ZRTextVerticalAlign;
    backgroundColor?: ZRColor;
    /**
     * Padding between text and border.
     * Support to be a single number or an array.
     */
    padding?: number | number[];
    /**
     * Gap between text and subtext
     */
    itemGap?: number;
    textStyle?: LabelOption;
    subtextStyle?: LabelOption;
    /**
     * If trigger mouse or touch event
     */
    triggerEvent?: boolean;
    /**
     * Radius of background border.
     */
    borderRadius?: number | number[];
}
export declare function install(registers: EChartsExtensionInstallRegisters): void;
