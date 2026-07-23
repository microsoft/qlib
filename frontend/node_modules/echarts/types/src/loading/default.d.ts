import { LoadingEffect } from '../util/types.js';
import ExtensionAPI from '../core/ExtensionAPI.js';
/**
 * @param {module:echarts/ExtensionAPI} api
 * @param {Object} [opts]
 * @param {string} [opts.text]
 * @param {string} [opts.color]
 * @param {string} [opts.textColor]
 * @return {module:zrender/Element}
 */
export default function defaultLoading(api: ExtensionAPI, opts?: {
    text?: string;
    color?: string;
    textColor?: string;
    maskColor?: string;
    zlevel?: number;
    showSpinner?: boolean;
    spinnerRadius?: number;
    lineWidth?: number;
    fontSize?: number;
    fontWeight?: 'normal' | 'bold' | 'bolder' | 'lighter' | number;
    fontStyle?: 'normal' | 'italic' | 'oblique';
    fontFamily?: string;
}): LoadingEffect;
