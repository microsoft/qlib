import Model from '../Model.js';
import { AreaStyleOption } from '../../util/types.js';
import { PathStyleProps } from 'zrender/lib/graphic/Path.js';
export declare const AREA_STYLE_KEY_MAP: string[][];
declare type AreaStyleProps = Pick<PathStyleProps, 'fill' | 'shadowBlur' | 'shadowOffsetX' | 'shadowOffsetY' | 'opacity' | 'shadowColor'>;
declare class AreaStyleMixin {
    getAreaStyle(this: Model, excludes?: readonly (keyof AreaStyleOption)[], includes?: readonly (keyof AreaStyleOption)[]): AreaStyleProps;
}
export { AreaStyleMixin };
