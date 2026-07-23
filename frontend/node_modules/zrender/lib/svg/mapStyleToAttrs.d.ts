import Path, { PathStyleProps } from '../graphic/Path';
import ZRImage, { ImageStyleProps } from '../graphic/Image';
import TSpan, { TSpanStyleProps } from '../graphic/TSpan';
declare type AllStyleOption = PathStyleProps | TSpanStyleProps | ImageStyleProps;
export default function mapStyleToAttrs(updateAttr: (key: string, val: string | number) => void, style: AllStyleOption, el: Path | TSpan | ZRImage, forceUpdate: boolean): void;
export {};
