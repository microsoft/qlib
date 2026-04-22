import * as zrUtil from 'zrender/lib/core/util.js';
import * as modelUtil from '../../util/model.js';
import { ComponentOption, BoxLayoutOptionMixin, Dictionary, ZRStyleProps, OptionId, CommonTooltipOption, AnimationOptionMixin, AnimationOption } from '../../util/types.js';
import ComponentModel from '../../model/Component.js';
import Element, { ElementTextConfig } from 'zrender/lib/Element.js';
import Displayable from 'zrender/lib/graphic/Displayable.js';
import { PathProps, PathStyleProps } from 'zrender/lib/graphic/Path.js';
import { ImageStyleProps, ImageProps } from 'zrender/lib/graphic/Image.js';
import { TextStyleProps, TextProps } from 'zrender/lib/graphic/Text.js';
import GlobalModel from '../../model/Global.js';
import { TransitionOptionMixin } from '../../animation/customGraphicTransition.js';
import { ElementKeyframeAnimationOption } from '../../animation/customGraphicKeyframeAnimation.js';
import { GroupProps } from 'zrender/lib/graphic/Group.js';
import { TransformProp } from 'zrender/lib/core/Transformable.js';
import { ElementEventNameWithOn } from 'zrender/lib/core/types.js';
interface GraphicComponentBaseElementOption extends Partial<Pick<Element, TransformProp | 'silent' | 'ignore' | 'textConfig' | 'draggable' | ElementEventNameWithOn>>, 
/**
 * left/right/top/bottom: (like 12, '22%', 'center', default undefined)
 * If left/right is set, shape.x/shape.cx/position will not be used.
 * If top/bottom is set, shape.y/shape.cy/position will not be used.
 * This mechanism is useful when you want to position a group/element
 * against the right side or the center of this container.
 */
Partial<Pick<BoxLayoutOptionMixin, 'left' | 'right' | 'top' | 'bottom'>> {
    /**
     * element type, mandatory.
     * Only can be omit if call setOption not at the first time and perform merge.
     */
    type?: string;
    id?: OptionId;
    name?: string;
    parentId?: OptionId;
    parentOption?: GraphicComponentElementOption;
    children?: GraphicComponentElementOption[];
    hv?: [boolean, boolean];
    /**
     * bounding: (enum: 'all' (default) | 'raw')
     * Specify how to calculate boundingRect when locating.
     * 'all': Get uioned and transformed boundingRect
     *     from both itself and its descendants.
     *     This mode simplies confining a group of elements in the bounding
     *     of their ancester container (e.g., using 'right: 0').
     * 'raw': Only use the boundingRect of itself and before transformed.
     *     This mode is similar to css behavior, which is useful when you
     *     want an element to be able to overflow its container. (Consider
     *     a rotated circle needs to be located in a corner.)
     */
    bounding?: 'raw' | 'all';
    /**
     * info: custom info. enables user to mount some info on elements and use them
     * in event handlers. Update them only when user specified, otherwise, remain.
     */
    info?: GraphicExtraElementInfo;
    clipPath?: Omit<GraphicComponentZRPathOption, 'clipPath'> | false;
    textContent?: Omit<GraphicComponentTextOption, 'clipPath'>;
    textConfig?: ElementTextConfig;
    $action?: 'merge' | 'replace' | 'remove';
    tooltip?: CommonTooltipOption<unknown>;
    enterAnimation?: AnimationOption;
    updateAnimation?: AnimationOption;
    leaveAnimation?: AnimationOption;
}
export interface GraphicComponentDisplayableOption extends GraphicComponentBaseElementOption, Partial<Pick<Displayable, 'zlevel' | 'z' | 'z2' | 'invisible' | 'cursor'>> {
    style?: ZRStyleProps;
    z2?: number;
}
export interface GraphicComponentGroupOption extends GraphicComponentBaseElementOption, TransitionOptionMixin<GroupProps> {
    type?: 'group';
    /**
     * width/height: (can only be pixel value, default 0)
     * Is only used to specify container (group) size, if needed. And
     * cannot be a percentage value (like '33%'). See the reason in the
     * layout algorithm below.
     */
    width?: number;
    height?: number;
    children: GraphicComponentElementOption[];
    keyframeAnimation?: ElementKeyframeAnimationOption<GroupProps> | ElementKeyframeAnimationOption<GroupProps>[];
}
export interface GraphicComponentZRPathOption extends GraphicComponentDisplayableOption, TransitionOptionMixin<PathProps> {
    shape?: PathProps['shape'] & TransitionOptionMixin<PathProps['shape']>;
    style?: PathStyleProps & TransitionOptionMixin<PathStyleProps>;
    keyframeAnimation?: ElementKeyframeAnimationOption<PathProps> | ElementKeyframeAnimationOption<PathProps>[];
}
export interface GraphicComponentImageOption extends GraphicComponentDisplayableOption, TransitionOptionMixin<ImageProps> {
    type?: 'image';
    style?: ImageStyleProps & TransitionOptionMixin<ImageStyleProps>;
    keyframeAnimation?: ElementKeyframeAnimationOption<ImageProps> | ElementKeyframeAnimationOption<ImageProps>[];
}
export interface GraphicComponentTextOption extends Omit<GraphicComponentDisplayableOption, 'textContent' | 'textConfig'>, TransitionOptionMixin<TextProps> {
    type?: 'text';
    style?: TextStyleProps & TransitionOptionMixin<TextStyleProps>;
    keyframeAnimation?: ElementKeyframeAnimationOption<TextProps> | ElementKeyframeAnimationOption<TextProps>[];
}
export declare type GraphicComponentElementOption = GraphicComponentGroupOption | GraphicComponentZRPathOption | GraphicComponentImageOption | GraphicComponentTextOption;
declare type GraphicExtraElementInfo = Dictionary<unknown>;
export declare type ElementMap = zrUtil.HashMap<Element, string>;
export declare type GraphicComponentLooseOption = (GraphicComponentOption | GraphicComponentElementOption) & {
    mainType?: 'graphic';
};
export interface GraphicComponentOption extends ComponentOption, AnimationOptionMixin {
    elements?: GraphicComponentElementOption[];
}
export declare function setKeyInfoToNewElOption(resultItem: ReturnType<typeof modelUtil.mappingToExists>[number], newElOption: GraphicComponentElementOption): void;
export declare class GraphicComponentModel extends ComponentModel<GraphicComponentOption> {
    static type: string;
    type: string;
    preventAutoZ: boolean;
    static defaultOption: GraphicComponentOption;
    /**
     * Save el options for the sake of the performance (only update modified graphics).
     * The order is the same as those in option. (ancesters -> descendants)
     */
    private _elOptionsToUpdate;
    mergeOption(option: GraphicComponentOption, ecModel: GlobalModel): void;
    optionUpdated(newOption: GraphicComponentOption, isInit: boolean): void;
    /**
     * Convert
     * [{
     *  type: 'group',
     *  id: 'xx',
     *  children: [{type: 'circle'}, {type: 'polygon'}]
     * }]
     * to
     * [
     *  {type: 'group', id: 'xx'},
     *  {type: 'circle', parentId: 'xx'},
     *  {type: 'polygon', parentId: 'xx'}
     * ]
     */
    private _flatten;
    useElOptionsToUpdate(): GraphicComponentElementOption[];
}
export {};
