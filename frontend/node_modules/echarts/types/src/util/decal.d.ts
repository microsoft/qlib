import { PatternObject } from 'zrender/lib/graphic/Pattern.js';
import ExtensionAPI from '../core/ExtensionAPI.js';
import { InnerDecalObject } from './types.js';
/**
 * Create or update pattern image from decal options
 *
 * @param {InnerDecalObject | 'none'} decalObject decal options, 'none' if no decal
 * @return {Pattern} pattern with generated image, null if no decal
 */
export declare function createOrUpdatePatternFromDecal(decalObject: InnerDecalObject | 'none', api: ExtensionAPI): PatternObject;
