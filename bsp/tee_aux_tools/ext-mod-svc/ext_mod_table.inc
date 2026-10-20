/* ext_mod_table.inc - 扩展模块表, 由 tee_ext_mod_svc.c 在编译期包含.
 *
 * 本文件由 bsp/sittim/modules/Makefile 的 mod-tbl 目标生成, 不手工编辑: 各项取部署目录 dist
 * 中的映像文件, 按模块名排序, 模块编号取自该模块源码中 MODULE_ID 的声明.
 *
 * 每项为一对 "<模块编号>, \"<映像文件名>\"". 映像文件名相对模块目录给出, 与
 * bsp/sittim/modules 的 dist 目标所产出的名称一致; 完整路径由取入方把模块目录与该
 * 文件名拼接得到, 故本文件不含目录.
 *
 * 模块编号与映像声明的编号不一致时飞地的取入以编号不符失败, 见 sittim 的
 * ext_mod::LoadError::IdMismatch; 宿主侧的 tee_ext_mod_serve 按请求中的编号查表, 查不到时
 * 以字节数 0 应答, 飞地据此判该模块不存在.
 */
{ 7, "attest.signed.bin" },
{ 3, "fs.signed.bin" },
{ 4, "net.signed.bin" },
