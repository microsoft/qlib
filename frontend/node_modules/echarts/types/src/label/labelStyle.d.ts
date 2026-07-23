import ZRText, { TextStyleProps } from 'zrender/lib/graphic/Text.js';
import Element, { ElementTextConfig } from 'zrender/lib/Element.js';
import Model from '../model/Model.js';
import { LabelOption, DisplayState, TextCommonOption, StatesOptionMixin, DisplayStateNonNormal, ColorString, ZRStyleProps, AnimationOptionMixin, InterpolatableValue } from '../util/types.js';
import GlobalModel from '../model/Global.js';
import SeriesData from '../data/SeriesData.js';
declare type TextCommonParams = {
    /**
     * Whether disable drawing box of block (outer most).
     */
    disableBox?: boolean;
    /**
     * Specify a color when color is 'inherit',
     * If inheritColor specified, it is used as default textFill.
     */
    inheritColor?: ColorString;
    /**
     * Specify a opacity when opacity is not given.
     */
    defaultOpacity?: number;
    defaultOutsidePosition?: LabelOption['position'];
    /**
     * If support legacy 'auto' for 'inherit' usage.
     */
    textStyle?: ZRStyleProps;
};
interface SetLabelStyleOpt<TLabelDataIndex> extends TextCommonParams {
    defaultText?: string | ((labelDataIndex: TLabelDataIndex, opt: SetLabelStyleOpt<TLabelDataIndex>, interpolatedValue?: InterpolatableValue) => string);
    labelFetcher?: {
        getFormattedLabel: (labelDataIndex: TLabelDataIndex, status: DisplayState, dataType?: string, labelDimIndex?: number, formatter?: string | ((params: object) => string), extendParams?: {
            interpolatedValue: InterpolatableValue;
        }) => string;
    };
    labelDataIndex?: TLabelDataIndex;
    labelDimIndex?: number;
    /**
     * Inject a setter of text for the text animation case.
     */
    enableTextSetter?: boolean;
}
declare type LabelModel = Model<LabelOption & {
    formatter?: string | ((params: any) => string);
    showDuringLabel?: boolean;
}>;
declare type LabelModelForText = Model<Omit<LabelOption, 'position' | 'rotate'> & {
    formatter?: string | ((params: any) => string);
}>;
declare type LabelStatesModels<LabelModel> = Partial<Record<DisplayStateNonNormal, LabelModel>> & {
    normal: LabelModel;
};
export declare function setLabelText(label: ZRText, labelTexts: Record<DisplayState, string>): void;
/**
 * Set normal styles and emphasis styles about text on target element
 * If target is a ZRText. It will create a new style object.
 * If target is other Element. It will create or reuse ZRText which is attached on the target.
 * And create a new style object.
 *
 * NOTICE: Because the style on ZRText will be replaced with new(only x, y are keeped).
 * So please update the style on ZRText after use this method.
 */
declare function setLabelStyle<TLabelDataIndex>(targetEl: ZRText, labelStatesModels: LabelStatesModels<LabelModelForText>, opt?: SetLabelStyleOpt<TLabelDataIndex>, stateSpecified?: Partial<Record<DisplayState, TextStyleProps>>): void;
declare function setLabelStyle<TLabelDataIndex>(targetEl: Element, labelStatesModels: LabelStatesModels<LabelModel>, opt?: SetLabelStyleOpt<TLabelDataIndex>, stateSpecified?: Partial<Record<DisplayState, TextStyleProps>>): void;
export { setLabelStyle };
export declare function getLabelStatesModels<LabelName extends string = 'label'>(itemModel: Model<StatesOptionMixin<any, any> & Partial<Record<LabelName, any>>>, labelName?: LabelName): Record<DisplayState, LabelModel>;
/**
 * Set basic textStyle properties.
 */
export declare function createTextStyle(textStyleModel: Model, specifiedTextStyle?: TextStyleProps, // Fixed style in the code. Can't be set by model.
opt?: Pick<TextCommonParams, 'inheritColor' | 'disableBox'>, isNotNormal?: boolean, isAttached?: boolean): TextStyleProps;
export declare function createTextConfig(textStyleModel: Model, opt?: Pick<TextCommonParams, 'defaultOutsidePosition' | 'inheritColor'>, isNotNormal?: boolean): ElementTextConfig;
export declare function getFont(opt: Pick<TextCommonOption, 'fontStyle' | 'fontWeight' | 'fontSize' | 'fontFamily'>, ecModel: GlobalModel): string;
export declare const labelInner: (hostObj: ZRText) => {
    /**
     * Previous target value stored used for label.
     * It's mainly for text animation
     */
    prevValue?: InterpolatableValue;
    /**
     * Target value stored used for label.
     */
    value?: InterpolatableValue;
    /**
     * Current value in text animation.
     */
    interpolatedValue?: InterpolatableValue;
    /**
     * If enable value animation
     */
    valueAnimation?: boolean;
    /**
     * Label value precision during animation.
     */
    precision?: number | 'auto';
    /**
     * If enable value animation
     */
    statesModels?: LabelStatesModels<LabelModelForText>;
    /**
     * Default text getter during interpolation
     */
    defaultInterpolatedText?: (value: InterpolatableValue) => string;
    /**
     * Change label text from interpolated text during animation
     */
    setLabelText?: (interpolatedValue?: InterpolatableValue) => void;
};
export declare function setLabelValueAnimation(label: ZRText, labelStatesModels: LabelStatesModels<LabelModelForText>, value: InterpolatableValue, getDefaultText: (value: InterpolatableValue) => string): void;
export declare function animateLabelValue(textEl: ZRText, dataIndex: number, data: SeriesData, animatableModel: Model<AnimationOptionMixin>, labelFetcher: SetLabelStyleOpt<number>['labelFetcher']): void;
