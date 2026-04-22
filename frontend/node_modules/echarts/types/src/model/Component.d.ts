import Model from './Model.js';
import * as componentUtil from '../util/component.js';
import { ExtendableConstructor, ClassManager } from '../util/clazz.js';
import { QueryReferringOpt } from '../util/model.js';
import GlobalModel from './Global.js';
import { ComponentOption, ComponentMainType, ComponentSubType, ComponentFullType, ComponentLayoutMode } from '../util/types.js';
declare class ComponentModel<Opt extends ComponentOption = ComponentOption> extends Model<Opt> {
    /**
     * @readonly
     */
    type: ComponentFullType;
    /**
     * @readonly
     */
    id: string;
    /**
     * Because simplified concept is probably better, series.name (or component.name)
     * has been having too many responsibilities:
     * (1) Generating id (which requires name in option should not be modified).
     * (2) As an index to mapping series when merging option or calling API (a name
     * can refer to more than one component, which is convenient is some cases).
     * (3) Display.
     * @readOnly But injected
     */
    name: string;
    /**
     * @readOnly
     */
    mainType: ComponentMainType;
    /**
     * @readOnly
     */
    subType: ComponentSubType;
    /**
     * @readOnly
     */
    componentIndex: number;
    /**
     * @readOnly
     */
    protected defaultOption: ComponentOption;
    /**
     * @readOnly
     */
    ecModel: GlobalModel;
    /**
     * @readOnly
     */
    static dependencies: string[];
    readonly uid: string;
    /**
     * Support merge layout params.
     * Only support 'box' now (left/right/top/bottom/width/height).
     */
    static layoutMode: ComponentLayoutMode | ComponentLayoutMode['type'];
    /**
     * Prevent from auto set z, zlevel, z2 by the framework.
     */
    preventAutoZ: boolean;
    __viewId: string;
    __requireNewView: boolean;
    static protoInitialize: void;
    constructor(option: Opt, parentModel: Model, ecModel: GlobalModel);
    init(option: Opt, parentModel: Model, ecModel: GlobalModel): void;
    mergeDefaultAndTheme(option: Opt, ecModel: GlobalModel): void;
    mergeOption(option: Opt, ecModel: GlobalModel): void;
    /**
     * Called immediately after `init` or `mergeOption` of this instance called.
     */
    optionUpdated(newCptOption: Opt, isInit: boolean): void;
    /**
     * [How to declare defaultOption]:
     *
     * (A) If using class declaration in typescript (since echarts 5):
     * ```ts
     * import {ComponentOption} from '../model/option.js';
     * export interface XxxOption extends ComponentOption {
     *     aaa: number
     * }
     * export class XxxModel extends Component {
     *     static type = 'xxx';
     *     static defaultOption: XxxOption = {
     *         aaa: 123
     *     }
     * }
     * Component.registerClass(XxxModel);
     * ```
     * ```ts
     * import {inheritDefaultOption} from '../util/component.js';
     * import {XxxModel, XxxOption} from './XxxModel.js';
     * export interface XxxSubOption extends XxxOption {
     *     bbb: number
     * }
     * class XxxSubModel extends XxxModel {
     *     static defaultOption: XxxSubOption = inheritDefaultOption(XxxModel.defaultOption, {
     *         bbb: 456
     *     })
     *     fn() {
     *         let opt = this.getDefaultOption();
     *         // opt is {aaa: 123, bbb: 456}
     *     }
     * }
     * ```
     *
     * (B) If using class extend (previous approach in echarts 3 & 4):
     * ```js
     * let XxxComponent = Component.extend({
     *     defaultOption: {
     *         xx: 123
     *     }
     * })
     * ```
     * ```js
     * let XxxSubComponent = XxxComponent.extend({
     *     defaultOption: {
     *         yy: 456
     *     },
     *     fn: function () {
     *         let opt = this.getDefaultOption();
     *         // opt is {xx: 123, yy: 456}
     *     }
     * })
     * ```
     */
    getDefaultOption(): Opt;
    /**
     * Notice: always force to input param `useDefault` in case that forget to consider it.
     * The same behavior as `modelUtil.parseFinder`.
     *
     * @param useDefault In many cases like series refer axis and axis refer grid,
     *        If axis index / axis id not specified, use the first target as default.
     *        In other cases like dataZoom refer axis, if not specified, measn no refer.
     */
    getReferringComponents(mainType: ComponentMainType, opt: QueryReferringOpt): {
        models: ComponentModel[];
        specified: boolean;
    };
    getBoxLayoutParams(): {
        left: string | number;
        top: string | number;
        right: string | number;
        bottom: string | number;
        width: string | number;
        height: string | number;
    };
    /**
     * Get key for zlevel.
     * If developers don't configure zlevel. We will assign zlevel to series based on the key.
     * For example, lines with trail effect and progressive series will in an individual zlevel.
     */
    getZLevelKey(): string;
    setZLevel(zlevel: number): void;
    static registerClass: ClassManager['registerClass'];
    static hasClass: ClassManager['hasClass'];
    static registerSubTypeDefaulter: componentUtil.SubTypeDefaulterManager['registerSubTypeDefaulter'];
}
export declare type ComponentModelConstructor = typeof ComponentModel & ClassManager & componentUtil.SubTypeDefaulterManager & ExtendableConstructor & componentUtil.TopologicalTravelable<object>;
export default ComponentModel;
