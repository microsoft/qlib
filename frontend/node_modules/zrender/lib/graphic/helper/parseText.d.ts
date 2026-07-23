import { TextAlign, TextVerticalAlign } from '../../core/types';
import { TextStyleProps } from '../Text';
interface InnerTruncateOption {
    maxIteration?: number;
    minChar?: number;
    placeholder?: string;
    maxIterations?: number;
}
export declare function truncateText(text: string, containerWidth: number, font: string, ellipsis?: string, options?: InnerTruncateOption): string;
export interface PlainTextContentBlock {
    lineHeight: number;
    calculatedLineHeight: number;
    contentWidth: number;
    contentHeight: number;
    width: number;
    height: number;
    outerWidth: number;
    outerHeight: number;
    lines: string[];
    isTruncated: boolean;
}
export declare function parsePlainText(text: string, style?: TextStyleProps): PlainTextContentBlock;
declare class RichTextToken {
    styleName: string;
    text: string;
    width: number;
    height: number;
    innerHeight: number;
    contentHeight: number;
    contentWidth: number;
    lineHeight: number;
    font: string;
    align: TextAlign;
    verticalAlign: TextVerticalAlign;
    textPadding: number[];
    percentWidth?: string;
    isLineHolder: boolean;
}
declare class RichTextLine {
    lineHeight: number;
    width: number;
    tokens: RichTextToken[];
    constructor(tokens?: RichTextToken[]);
}
export declare class RichTextContentBlock {
    width: number;
    height: number;
    contentWidth: number;
    contentHeight: number;
    outerWidth: number;
    outerHeight: number;
    lines: RichTextLine[];
    isTruncated: boolean;
}
export declare function parseRichText(text: string, style: TextStyleProps): RichTextContentBlock;
export {};
