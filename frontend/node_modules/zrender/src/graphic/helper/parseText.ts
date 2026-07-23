import * as imageHelper from '../helper/image';
import {
    extend,
    retrieve2,
    retrieve3,
    reduce
} from '../../core/util';
import { TextAlign, TextVerticalAlign, ImageLike, Dictionary } from '../../core/types';
import { TextStyleProps } from '../Text';
import { getLineHeight, getWidth, parsePercent } from '../../contain/text';

const STYLE_REG = /\{([a-zA-Z0-9_]+)\|([^}]*)\}/g;

interface InnerTruncateOption {
    maxIteration?: number
    // If truncate result are less than minChar, ellipsis will not show
    // which is better for user hint in some cases
    minChar?: number
    // When all truncated, use the placeholder
    placeholder?: string

    maxIterations?: number
}

interface InnerPreparedTruncateOption extends Required<InnerTruncateOption> {
    font: string

    ellipsis: string
    ellipsisWidth: number
    contentWidth: number

    containerWidth: number
    cnCharWidth: number
    ascCharWidth: number
}

/**
 * Show ellipsis if overflow.
 */
export function truncateText(
    text: string,
    containerWidth: number,
    font: string,
    ellipsis?: string,
    options?: InnerTruncateOption
): string {
    const out = {} as Parameters<typeof truncateText2>[0];
    truncateText2(out, text, containerWidth, font, ellipsis, options);
    return out.text;
}

// PENDING: not sure whether `truncateText` is used outside zrender, since it has an `export`
// specifier. So keep it and perform the interface modification in `truncateText2`.
function truncateText2(
    out: {text: string, isTruncated: boolean},
    text: string,
    containerWidth: number,
    font: string,
    ellipsis?: string,
    options?: InnerTruncateOption
): void {
    if (!containerWidth) {
        out.text = '';
        out.isTruncated = false;
        return;
    }

    const textLines = (text + '').split('\n');
    options = prepareTruncateOptions(containerWidth, font, ellipsis, options);

    // FIXME
    // It is not appropriate that every line has '...' when truncate multiple lines.
    let isTruncated = false;
    const truncateOut = {} as Parameters<typeof truncateSingleLine>[0];
    for (let i = 0, len = textLines.length; i < len; i++) {
        truncateSingleLine(truncateOut, textLines[i], options as InnerPreparedTruncateOption);
        textLines[i] = truncateOut.textLine;
        isTruncated = isTruncated || truncateOut.isTruncated;
    }

    out.text = textLines.join('\n');
    out.isTruncated = isTruncated;
}

function prepareTruncateOptions(
    containerWidth: number,
    font: string,
    ellipsis?: string,
    options?: InnerTruncateOption
): InnerPreparedTruncateOption {
    options = options || {};
    let preparedOpts = extend({}, options) as InnerPreparedTruncateOption;

    preparedOpts.font = font;
    ellipsis = retrieve2(ellipsis, '...');
    preparedOpts.maxIterations = retrieve2(options.maxIterations, 2);
    const minChar = preparedOpts.minChar = retrieve2(options.minChar, 0);
    // FIXME
    // Other languages?
    preparedOpts.cnCharWidth = getWidth('国', font);
    // FIXME
    // Consider proportional font?
    const ascCharWidth = preparedOpts.ascCharWidth = getWidth('a', font);
    preparedOpts.placeholder = retrieve2(options.placeholder, '');

    // Example 1: minChar: 3, text: 'asdfzxcv', truncate result: 'asdf', but not: 'a...'.
    // Example 2: minChar: 3, text: '维度', truncate result: '维', but not: '...'.
    let contentWidth = containerWidth = Math.max(0, containerWidth - 1); // Reserve some gap.
    for (let i = 0; i < minChar && contentWidth >= ascCharWidth; i++) {
        contentWidth -= ascCharWidth;
    }

    let ellipsisWidth = getWidth(ellipsis, font);
    if (ellipsisWidth > contentWidth) {
        ellipsis = '';
        ellipsisWidth = 0;
    }

    contentWidth = containerWidth - ellipsisWidth;

    preparedOpts.ellipsis = ellipsis;
    preparedOpts.ellipsisWidth = ellipsisWidth;
    preparedOpts.contentWidth = contentWidth;
    preparedOpts.containerWidth = containerWidth;

    return preparedOpts;
}

function truncateSingleLine(
    out: {textLine: string, isTruncated: boolean},
    textLine: string,
    options: InnerPreparedTruncateOption
): void {
    const containerWidth = options.containerWidth;
    const font = options.font;
    const contentWidth = options.contentWidth;

    if (!containerWidth) {
        out.textLine = '';
        out.isTruncated = false;
        return;
    }

    let lineWidth = getWidth(textLine, font);

    if (lineWidth <= containerWidth) {
        out.textLine = textLine;
        out.isTruncated = false;
        return;
    }

    for (let j = 0; ; j++) {
        if (lineWidth <= contentWidth || j >= options.maxIterations) {
            textLine += options.ellipsis;
            break;
        }

        const subLength = j === 0
            ? estimateLength(textLine, contentWidth, options.ascCharWidth, options.cnCharWidth)
            : lineWidth > 0
            ? Math.floor(textLine.length * contentWidth / lineWidth)
            : 0;

        textLine = textLine.substr(0, subLength);
        lineWidth = getWidth(textLine, font);
    }

    if (textLine === '') {
        textLine = options.placeholder;
    }

    out.textLine = textLine;
    out.isTruncated = true;
}

function estimateLength(
    text: string, contentWidth: number, ascCharWidth: number, cnCharWidth: number
): number {
    let width = 0;
    let i = 0;
    for (let len = text.length; i < len && width < contentWidth; i++) {
        const charCode = text.charCodeAt(i);
        width += (0 <= charCode && charCode <= 127) ? ascCharWidth : cnCharWidth;
    }
    return i;
}

export interface PlainTextContentBlock {
    lineHeight: number
    // Line height of actual content.
    calculatedLineHeight: number

    contentWidth: number
    contentHeight: number

    width: number
    height: number

    /**
     * Real text width containing padding.
     * It should be the same as `width` if background is rendered
     * and `width` is set by user.
     */
    outerWidth: number
    outerHeight: number

    lines: string[]

    // Be `true` if and only if the result text is modified due to overflow, due to
    // settings on either `overflow` or `lineOverflow`
    isTruncated: boolean
}

export function parsePlainText(
    text: string,
    style?: TextStyleProps
): PlainTextContentBlock {
    text != null && (text += '');

    // textPadding has been normalized
    const overflow = style.overflow;
    const padding = style.padding as number[];
    const font = style.font;
    const truncate = overflow === 'truncate';
    const calculatedLineHeight = getLineHeight(font);
    const lineHeight = retrieve2(style.lineHeight, calculatedLineHeight);
    const bgColorDrawn = !!(style.backgroundColor);

    const truncateLineOverflow = style.lineOverflow === 'truncate';
    let isTruncated = false;

    let width = style.width;
    let lines: string[];

    if (width != null && (overflow === 'break' || overflow === 'breakAll')) {
        lines = text ? wrapText(text, style.font, width, overflow === 'breakAll', 0).lines : [];
    }
    else {
        lines = text ? text.split('\n') : [];
    }

    const contentHeight = lines.length * lineHeight;
    const height = retrieve2(style.height, contentHeight);

    // Truncate lines.
    if (contentHeight > height && truncateLineOverflow) {
        const lineCount = Math.floor(height / lineHeight);

        isTruncated = isTruncated || (lines.length > lineCount);
        lines = lines.slice(0, lineCount);

        // TODO If show ellipse for line truncate
        // if (style.ellipsis) {
        //     const options = prepareTruncateOptions(width, font, style.ellipsis, {
        //         minChar: style.truncateMinChar,
        //         placeholder: style.placeholder
        //     });
        //     lines[lineCount - 1] = truncateSingleLine(lastLine, options);
        // }
    }

    if (text && truncate && width != null) {
        const options = prepareTruncateOptions(width, font, style.ellipsis, {
            minChar: style.truncateMinChar,
            placeholder: style.placeholder
        });
        // Having every line has '...' when truncate multiple lines.
        const singleOut = {} as Parameters<typeof truncateSingleLine>[0];
        for (let i = 0; i < lines.length; i++) {
            truncateSingleLine(singleOut, lines[i], options);
            lines[i] = singleOut.textLine;
            isTruncated = isTruncated || singleOut.isTruncated;
        }
    }

    // Calculate real text width and height
    let outerHeight = height;
    let contentWidth = 0;
    for (let i = 0; i < lines.length; i++) {
        contentWidth = Math.max(getWidth(lines[i], font), contentWidth);
    }
    if (width == null) {
        // When width is not explicitly set, use outerWidth as width.
        width = contentWidth;
    }

    let outerWidth = contentWidth;
    if (padding) {
        outerHeight += padding[0] + padding[2];
        outerWidth += padding[1] + padding[3];
        width += padding[1] + padding[3];
    }

    if (bgColorDrawn) {
        // When render background, outerWidth should be the same as width.
        outerWidth = width;
    }

    return {
        lines: lines,
        height: height,
        outerWidth: outerWidth,
        outerHeight: outerHeight,
        lineHeight: lineHeight,
        calculatedLineHeight: calculatedLineHeight,
        contentWidth: contentWidth,
        contentHeight: contentHeight,
        width: width,
        isTruncated: isTruncated
    };
}

class RichTextToken {
    styleName: string
    text: string
    width: number
    height: number

    // Inner height exclude padding
    innerHeight: number

    // Width and height of actual text content.
    contentHeight: number
    contentWidth: number

    lineHeight: number
    font: string
    align: TextAlign
    verticalAlign: TextVerticalAlign

    textPadding: number[]
    percentWidth?: string

    isLineHolder: boolean
}
class RichTextLine {
    lineHeight: number
    width: number
    tokens: RichTextToken[] = []

    constructor(tokens?: RichTextToken[]) {
        if (tokens) {
            this.tokens = tokens;
        }
    }
}
export class RichTextContentBlock {
    // width/height of content
    width: number = 0
    height: number = 0
    // Calculated text height
    contentWidth: number = 0
    contentHeight: number = 0
    // outerWidth/outerHeight with padding
    outerWidth: number = 0
    outerHeight: number = 0
    lines: RichTextLine[] = []
    // Be `true` if and only if the result text is modified due to overflow, due to
    // settings on either `overflow` or `lineOverflow`
    isTruncated: boolean = false
}

type WrapInfo = {
    width: number,
    accumWidth: number,
    breakAll: boolean
}
/**
 * For example: 'some text {a|some text}other text{b|some text}xxx{c|}xxx'
 * Also consider 'bbbb{a|xxx\nzzz}xxxx\naaaa'.
 * If styleName is undefined, it is plain text.
 */
export function parseRichText(text: string, style: TextStyleProps): RichTextContentBlock {
    const contentBlock = new RichTextContentBlock();

    text != null && (text += '');
    if (!text) {
        return contentBlock;
    }

    const topWidth = style.width;
    const topHeight = style.height;
    const overflow = style.overflow;
    let wrapInfo: WrapInfo = (overflow === 'break' || overflow === 'breakAll') && topWidth != null
        ? {width: topWidth, accumWidth: 0, breakAll: overflow === 'breakAll'}
        : null;

    let lastIndex = STYLE_REG.lastIndex = 0;
    let result;
    while ((result = STYLE_REG.exec(text)) != null) {
        const matchedIndex = result.index;
        if (matchedIndex > lastIndex) {
            pushTokens(contentBlock, text.substring(lastIndex, matchedIndex), style, wrapInfo);
        }
        pushTokens(contentBlock, result[2], style, wrapInfo, result[1]);
        lastIndex = STYLE_REG.lastIndex;
    }

    if (lastIndex < text.length) {
        pushTokens(contentBlock, text.substring(lastIndex, text.length), style, wrapInfo);
    }

    // For `textWidth: xx%`
    let pendingList = [];

    let calculatedHeight = 0;
    let calculatedWidth = 0;

    const stlPadding = style.padding as number[];

    const truncate = overflow === 'truncate';
    const truncateLine = style.lineOverflow === 'truncate';
    const tmpTruncateOut = {} as Parameters<typeof truncateText2>[0];

    // let prevToken: RichTextToken;

    function finishLine(line: RichTextLine, lineWidth: number, lineHeight: number) {
        line.width = lineWidth;
        line.lineHeight = lineHeight;
        calculatedHeight += lineHeight;
        calculatedWidth = Math.max(calculatedWidth, lineWidth);
    }
    // Calculate layout info of tokens.
    outer: for (let i = 0; i < contentBlock.lines.length; i++) {
        const line = contentBlock.lines[i];
        let lineHeight = 0;
        let lineWidth = 0;

        for (let j = 0; j < line.tokens.length; j++) {
            const token = line.tokens[j];
            const tokenStyle = token.styleName && style.rich[token.styleName] || {};
            // textPadding should not inherit from style.
            const textPadding = token.textPadding = tokenStyle.padding as number[];
            const paddingH = textPadding ? textPadding[1] + textPadding[3] : 0;

            const font = token.font = tokenStyle.font || style.font;

            token.contentHeight = getLineHeight(font);
            // textHeight can be used when textVerticalAlign is specified in token.
            let tokenHeight = retrieve2(
                // textHeight should not be inherited, consider it can be specified
                // as box height of the block.
                tokenStyle.height, token.contentHeight
            );
            token.innerHeight = tokenHeight;

            textPadding && (tokenHeight += textPadding[0] + textPadding[2]);
            token.height = tokenHeight;
            // Inlcude padding in lineHeight.
            token.lineHeight = retrieve3(
                tokenStyle.lineHeight, style.lineHeight, tokenHeight
            );

            token.align = tokenStyle && tokenStyle.align || style.align;
            token.verticalAlign = tokenStyle && tokenStyle.verticalAlign || 'middle';

            if (truncateLine && topHeight != null && calculatedHeight + token.lineHeight > topHeight) {
                // TODO Add ellipsis on the previous token.
                // prevToken.text =
                const originalLength = contentBlock.lines.length;
                if (j > 0) {
                    line.tokens = line.tokens.slice(0, j);
                    finishLine(line, lineWidth, lineHeight);
                    contentBlock.lines = contentBlock.lines.slice(0, i + 1);
                }
                else {
                    contentBlock.lines = contentBlock.lines.slice(0, i);
                }
                contentBlock.isTruncated = contentBlock.isTruncated || (contentBlock.lines.length < originalLength);
                break outer;
            }

            let styleTokenWidth = tokenStyle.width;
            let tokenWidthNotSpecified = styleTokenWidth == null || styleTokenWidth === 'auto';

            // Percent width, can be `100%`, can be used in drawing separate
            // line when box width is needed to be auto.
            if (typeof styleTokenWidth === 'string' && styleTokenWidth.charAt(styleTokenWidth.length - 1) === '%') {
                token.percentWidth = styleTokenWidth;
                pendingList.push(token);

                token.contentWidth = getWidth(token.text, font);
                // Do not truncate in this case, because there is no user case
                // and it is too complicated.
            }
            else {
                if (tokenWidthNotSpecified) {
                    // FIXME: If image is not loaded and textWidth is not specified, calling
                    // `getBoundingRect()` will not get correct result.
                    const textBackgroundColor = tokenStyle.backgroundColor;
                    let bgImg = textBackgroundColor && (textBackgroundColor as { image: ImageLike }).image;

                    if (bgImg) {
                        bgImg = imageHelper.findExistImage(bgImg);
                        if (imageHelper.isImageReady(bgImg)) {
                            // Update token width from image size.
                            token.width = Math.max(token.width, bgImg.width * tokenHeight / bgImg.height);
                        }
                    }
                }

                const remainTruncWidth = truncate && topWidth != null
                    ? topWidth - lineWidth : null;

                if (remainTruncWidth != null && remainTruncWidth < token.width) {
                    if (!tokenWidthNotSpecified || remainTruncWidth < paddingH) {
                        token.text = '';
                        token.width = token.contentWidth = 0;
                    }
                    else {
                        truncateText2(
                            tmpTruncateOut,
                            token.text, remainTruncWidth - paddingH, font, style.ellipsis,
                            {minChar: style.truncateMinChar}
                        );
                        token.text = tmpTruncateOut.text;
                        contentBlock.isTruncated = contentBlock.isTruncated || tmpTruncateOut.isTruncated;
                        token.width = token.contentWidth = getWidth(token.text, font);
                    }
                }
                else {
                    token.contentWidth = getWidth(token.text, font);
                }
            }

            token.width += paddingH;

            lineWidth += token.width;
            tokenStyle && (lineHeight = Math.max(lineHeight, token.lineHeight));

            // prevToken = token;
        }

        finishLine(line, lineWidth, lineHeight);
    }

    contentBlock.outerWidth = contentBlock.width = retrieve2(topWidth, calculatedWidth);
    contentBlock.outerHeight = contentBlock.height = retrieve2(topHeight, calculatedHeight);
    contentBlock.contentHeight = calculatedHeight;
    contentBlock.contentWidth = calculatedWidth;

    if (stlPadding) {
        contentBlock.outerWidth += stlPadding[1] + stlPadding[3];
        contentBlock.outerHeight += stlPadding[0] + stlPadding[2];
    }

    for (let i = 0; i < pendingList.length; i++) {
        const token = pendingList[i];
        const percentWidth = token.percentWidth;
        // Should not base on outerWidth, because token can not be placed out of padding.
        token.width = parseInt(percentWidth, 10) / 100 * contentBlock.width;
    }

    return contentBlock;
}

type TokenStyle = TextStyleProps['rich'][string];

function pushTokens(
    block: RichTextContentBlock,
    str: string,
    style: TextStyleProps,
    wrapInfo: WrapInfo,
    styleName?: string
) {
    const isEmptyStr = str === '';
    const tokenStyle: TokenStyle = styleName && style.rich[styleName] || {};
    const lines = block.lines;
    const font = tokenStyle.font || style.font;
    let newLine = false;
    let strLines;
    let linesWidths;

    if (wrapInfo) {
        const tokenPadding = tokenStyle.padding as number[];
        let tokenPaddingH = tokenPadding ? tokenPadding[1] + tokenPadding[3] : 0;
        if (tokenStyle.width != null && tokenStyle.width !== 'auto') {
            // Wrap the whole token if tokenWidth if fixed.
            const outerWidth = parsePercent(tokenStyle.width, wrapInfo.width) + tokenPaddingH;
            if (lines.length > 0) { // Not first line
                if (outerWidth + wrapInfo.accumWidth > wrapInfo.width) {
                    // TODO Support wrap text in token.
                    strLines = str.split('\n');
                    newLine = true;
                }
            }

            wrapInfo.accumWidth = outerWidth;
        }
        else {
            const res = wrapText(str, font, wrapInfo.width, wrapInfo.breakAll, wrapInfo.accumWidth);
            wrapInfo.accumWidth = res.accumWidth + tokenPaddingH;
            linesWidths = res.linesWidths;
            strLines = res.lines;
        }
    }
    else {
        strLines = str.split('\n');
    }

    for (let i = 0; i < strLines.length; i++) {
        const text = strLines[i];
        const token = new RichTextToken();
        token.styleName = styleName;
        token.text = text;
        token.isLineHolder = !text && !isEmptyStr;

        if (typeof tokenStyle.width === 'number') {
            token.width = tokenStyle.width;
        }
        else {
            token.width = linesWidths
                ? linesWidths[i] // Caculated width in the wrap
                : getWidth(text, font);
        }

        // The first token should be appended to the last line if not new line.
        if (!i && !newLine) {
            const tokens = (lines[lines.length - 1] || (lines[0] = new RichTextLine())).tokens;

            // Consider cases:
            // (1) ''.split('\n') => ['', '\n', ''], the '' at the first item
            // (which is a placeholder) should be replaced by new token.
            // (2) A image backage, where token likes {a|}.
            // (3) A redundant '' will affect textAlign in line.
            // (4) tokens with the same tplName should not be merged, because
            // they should be displayed in different box (with border and padding).
            const tokensLen = tokens.length;
            (tokensLen === 1 && tokens[0].isLineHolder)
                ? (tokens[0] = token)
                // Consider text is '', only insert when it is the "lineHolder" or
                // "emptyStr". Otherwise a redundant '' will affect textAlign in line.
                : ((text || !tokensLen || isEmptyStr) && tokens.push(token));
        }
        // Other tokens always start a new line.
        else {
            // If there is '', insert it as a placeholder.
            lines.push(new RichTextLine([token]));
        }
    }
}


function isAlphabeticLetter(ch: string) {
    // Unicode Character Ranges
    // https://jrgraphix.net/research/unicode_blocks.php
    // The following ranges may not cover all letter ranges but only the more
    // popular ones. Developers could make pull requests when they find those
    // not covered.
    let code = ch.charCodeAt(0);
    return code >= 0x20 && code <= 0x24F // Latin
        || code >= 0x370 && code <= 0x10FF // Greek, Coptic, Cyrilic, and etc.
        || code >= 0x1200 && code <= 0x13FF // Ethiopic and Cherokee
        || code >= 0x1E00 && code <= 0x206F; // Latin and Greek extended
}

const breakCharMap = reduce(',&?/;] '.split(''), function (obj, ch) {
    obj[ch] = true;
    return obj;
}, {} as Dictionary<boolean>);
/**
 * If break by word. For latin languages.
 */
function isWordBreakChar(ch: string) {
    if (isAlphabeticLetter(ch)) {
        if (breakCharMap[ch]) {
            return true;
        }
        return false;
    }
    return true;
}

function wrapText(
    text: string,
    font: string,
    lineWidth: number,
    isBreakAll: boolean,
    lastAccumWidth: number
) {
    let lines: string[] = [];
    let linesWidths: number[] = [];
    let line = '';
    let currentWord = '';
    let currentWordWidth = 0;
    let accumWidth = 0;

    for (let i = 0; i < text.length; i++) {

        const ch = text.charAt(i);
        if (ch === '\n') {
            if (currentWord) {
                line += currentWord;
                accumWidth += currentWordWidth;
            }
            lines.push(line);
            linesWidths.push(accumWidth);
            // Reset
            line = '';
            currentWord = '';
            currentWordWidth = 0;
            accumWidth = 0;
            continue;
        }

        const chWidth = getWidth(ch, font);
        const inWord = isBreakAll ? false : !isWordBreakChar(ch);

        if (!lines.length
            ? lastAccumWidth + accumWidth + chWidth > lineWidth
            : accumWidth + chWidth > lineWidth
        ) {
            if (!accumWidth) {  // If nothing appended yet.
                if (inWord) {
                    // The word length is still too long for one line
                    // Force break the word
                    lines.push(currentWord);
                    linesWidths.push(currentWordWidth);

                    currentWord = ch;
                    currentWordWidth = chWidth;
                }
                else {
                    // lineWidth is too small for ch
                    lines.push(ch);
                    linesWidths.push(chWidth);
                }
            }
            else if (line || currentWord) {
                if (inWord) {
                    if (!line) {
                        // The one word is still too long for one line
                        // Force break the word
                        // TODO Keep the word?
                        line = currentWord;
                        currentWord = '';
                        currentWordWidth = 0;
                        accumWidth = currentWordWidth;
                    }

                    lines.push(line);
                    linesWidths.push(accumWidth - currentWordWidth);

                    // Break the whole word
                    currentWord += ch;
                    currentWordWidth += chWidth;
                    line = '';
                    accumWidth = currentWordWidth;
                }
                else {
                    // Append lastWord if have
                    if (currentWord) {
                        line += currentWord;
                        currentWord = '';
                        currentWordWidth = 0;
                    }
                    lines.push(line);
                    linesWidths.push(accumWidth);

                    line = ch;
                    accumWidth = chWidth;
                }
            }

            continue;
        }

        accumWidth += chWidth;

        if (inWord) {
            currentWord += ch;
            currentWordWidth += chWidth;
        }
        else {
            // Append whole word
            if (currentWord) {
                line += currentWord;
                // Reset
                currentWord = '';
                currentWordWidth = 0;
            }

            // Append character
            line += ch;
        }
    }

    if (!lines.length && !line) {
        line = text;
        currentWord = '';
        currentWordWidth = 0;
    }

    // Append last line.
    if (currentWord) {
        line += currentWord;
    }
    if (line) {
        lines.push(line);
        linesWidths.push(accumWidth);
    }

    if (lines.length === 1) {
        // No new line.
        accumWidth += lastAccumWidth;
    }

    return {
        // Accum width of last line
        accumWidth,
        lines: lines,
        linesWidths
    };
}