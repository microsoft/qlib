import ZRText from 'zrender/lib/graphic/Text.js';
import { LabelLayoutOption } from '../util/types.js';
import { BoundingRect, OrientedBoundingRect, Polyline } from '../util/graphic.js';
interface LabelLayoutListPrepareInput {
    label: ZRText;
    labelLine?: Polyline;
    computedLayoutOption?: LabelLayoutOption;
    priority: number;
    defaultAttr: {
        ignore: boolean;
        labelGuideIgnore?: boolean;
    };
}
export interface LabelLayoutInfo {
    label: ZRText;
    labelLine: Polyline;
    priority: number;
    rect: BoundingRect;
    localRect: BoundingRect;
    obb?: OrientedBoundingRect;
    axisAligned: boolean;
    layoutOption: LabelLayoutOption;
    defaultAttr: {
        ignore: boolean;
        labelGuideIgnore?: boolean;
    };
    transform: number[];
}
export declare function prepareLayoutList(input: LabelLayoutListPrepareInput[]): LabelLayoutInfo[];
/**
 * Adjust labels on x direction to avoid overlap.
 */
export declare function shiftLayoutOnX(list: Pick<LabelLayoutInfo, 'rect' | 'label'>[], leftBound: number, rightBound: number, balanceShift?: boolean): boolean;
/**
 * Adjust labels on y direction to avoid overlap.
 */
export declare function shiftLayoutOnY(list: Pick<LabelLayoutInfo, 'rect' | 'label'>[], topBound: number, bottomBound: number, balanceShift?: boolean): boolean;
export declare function hideOverlap(labelList: LabelLayoutInfo[]): void;
export {};
