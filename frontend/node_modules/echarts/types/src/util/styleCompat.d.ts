import { ZRStyleProps } from './types.js';
import { ElementTextConfig } from 'zrender/lib/Element.js';
import { TextStyleProps, TextProps } from 'zrender/lib/graphic/Text.js';
import { ItemStyleProps } from '../model/mixin/itemStyle.js';
export interface LegacyStyleProps {
    legacy?: boolean;
}
/**
 * Whether need to call `convertEC4CompatibleStyle`.
 */
export declare function isEC4CompatibleStyle(style: ZRStyleProps & LegacyStyleProps, elType: string, hasOwnTextContentOption: boolean, hasOwnTextConfig: boolean): boolean;
/**
 * `EC4CompatibleStyle` is style that might be in echarts4 format or echarts5 format.
 * @param hostStyle The properties might be modified.
 * @return If be text el, `textContentStyle` and `textConfig` will not be returned.
 *         Otherwise a `textContentStyle` and `textConfig` will be created, whose props area
 *         retried from the `hostStyle`.
 */
export declare function convertFromEC4CompatibleStyle(hostStyle: ZRStyleProps, elType: string, isNormal: boolean): {
    textContent: TextProps & {
        type: string;
    };
    textConfig: ElementTextConfig;
};
/**
 * Convert to pure echarts4 format style.
 * `itemStyle` will be modified, added with ec4 style properties from
 * `textStyle` and `textConfig`.
 *
 * [Caveat]: For simplicity, `insideRollback` in ec4 does not compat, where
 * `styleEmphasis: {textFill: 'red'}` will remove the normal auto added stroke.
 */
export declare function convertToEC4StyleForCustomSerise(itemStl: ItemStyleProps, txStl: TextStyleProps, txCfg: ElementTextConfig): ZRStyleProps;
export declare function warnDeprecated(deprecated: string, insteadApproach: string): void;
