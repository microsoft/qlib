import * as graphicUtil from '../../util/graphic.js';
import Model from '../Model.js';
import { LabelOption, ColorString } from '../../util/types.js';
export declare type LabelFontOption = Pick<LabelOption, 'fontStyle' | 'fontWeight' | 'fontSize' | 'fontFamily'>;
declare type LabelRectRelatedOption = Pick<LabelOption, 'align' | 'verticalAlign' | 'padding' | 'lineHeight' | 'baseline' | 'rich' | 'width' | 'height' | 'overflow'> & LabelFontOption;
declare class TextStyleMixin {
    /**
     * Get color property or get color from option.textStyle.color
     */
    getTextColor(this: Model, isEmphasis?: boolean): ColorString;
    /**
     * Create font string from fontStyle, fontWeight, fontSize, fontFamily
     * @return {string}
     */
    getFont(this: Model<LabelFontOption>): string;
    getTextRect(this: Model<LabelRectRelatedOption> & TextStyleMixin, text: string): graphicUtil.BoundingRect;
}
export default TextStyleMixin;
