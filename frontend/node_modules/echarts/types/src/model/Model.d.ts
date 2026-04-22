import { AreaStyleMixin } from './mixin/areaStyle.js';
import TextStyleMixin from './mixin/textStyle.js';
import { LineStyleMixin } from './mixin/lineStyle.js';
import { ItemStyleMixin } from './mixin/itemStyle.js';
import GlobalModel from './Global.js';
import { ModelOption } from '../util/types.js';
interface Model<Opt = ModelOption> extends LineStyleMixin, ItemStyleMixin, TextStyleMixin, AreaStyleMixin {
}
declare class Model<Opt = ModelOption> {
    parentModel: Model;
    ecModel: GlobalModel;
    option: Opt;
    constructor(option?: Opt, parentModel?: Model, ecModel?: GlobalModel);
    init(option: Opt, parentModel?: Model, ecModel?: GlobalModel, ...rest: any): void;
    /**
     * Merge the input option to me.
     */
    mergeOption(option: Opt, ecModel?: GlobalModel): void;
    get<R extends keyof Opt>(path: R, ignoreParent?: boolean): Opt[R];
    get<R extends keyof Opt>(path: readonly [R], ignoreParent?: boolean): Opt[R];
    get<R extends keyof Opt, S extends keyof Opt[R]>(path: readonly [R, S], ignoreParent?: boolean): Opt[R][S];
    get<R extends keyof Opt, S extends keyof Opt[R], T extends keyof Opt[R][S]>(path: readonly [R, S, T], ignoreParent?: boolean): Opt[R][S][T];
    getShallow<R extends keyof Opt>(key: R, ignoreParent?: boolean): Opt[R];
    getModel<R extends keyof Opt>(path: R, parentModel?: Model): Model<Opt[R]>;
    getModel<R extends keyof Opt>(path: readonly [R], parentModel?: Model): Model<Opt[R]>;
    getModel<R extends keyof Opt, S extends keyof Opt[R]>(path: readonly [R, S], parentModel?: Model): Model<Opt[R][S]>;
    getModel<Ra extends keyof Opt, Rb extends keyof Opt, S extends keyof Opt[Rb]>(path: readonly [Ra] | readonly [Rb, S], parentModel?: Model): Model<Opt[Ra]> | Model<Opt[Rb][S]>;
    getModel<R extends keyof Opt, S extends keyof Opt[R], T extends keyof Opt[R][S]>(path: readonly [R, S, T], parentModel?: Model): Model<Opt[R][S][T]>;
    /**
     * If model has option
     */
    isEmpty(): boolean;
    restoreData(): void;
    clone(): Model<Opt>;
    parsePath(path: string | readonly string[]): readonly string[];
    resolveParentPath(path: readonly string[]): string[];
    isAnimationEnabled(): boolean;
    private _doGet;
}
export default Model;
