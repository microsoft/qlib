import Displayable from '../graphic/Displayable';
import { SVGVNodeAttrs, BrushScope } from './core';
export declare const EASING_MAP: Record<string, string>;
export declare const ANIMATE_STYLE_MAP: Record<string, string>;
export declare function createCSSAnimation(el: Displayable, attrs: SVGVNodeAttrs, scope: BrushScope, onlyShape?: boolean): void;
