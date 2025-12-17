use std::collections::HashMap;

use darling::FromMeta;
use syn::{FnArg, Pat, Type};

#[derive(Debug, FromMeta)]
pub(crate) struct ToolArgs {
    #[darling(default)]
    pub(crate) description: Option<String>,
    #[darling(default, rename = "args")]
    pub(crate) arg_descriptions: HashMap<String, String>,
}

/// 参数信息提取
pub(crate) struct ParamInfo {
    pub(crate) name: String,
    pub(crate) ty: Type,
    pub(crate) is_optional: bool,
}

pub(crate) fn extract_params(fn_args: &[FnArg]) -> Result<Vec<ParamInfo>, darling::Error> {
    let mut params = Vec::new();

    for arg in fn_args {
        match arg {
            FnArg::Typed(pat_type) => {
                let name = match &*pat_type.pat {
                    Pat::Ident(ident_pat) => ident_pat.ident.to_string(),
                    _ => {
                        return Err(darling::Error::custom(
                            "parameter must be a simple identifier",
                        ));
                    }
                };

                let ty = &*pat_type.ty;
                let is_optional = is_option_type(ty);

                params.push(ParamInfo {
                    name,
                    ty: ty.clone(),
                    is_optional,
                });
            }
            FnArg::Receiver(_) => {
                return Err(darling::Error::custom(
                    "tool functions cannot have self parameters",
                ));
            }
        }
    }

    Ok(params)
}

fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            segment.ident == "Option"
        } else {
            false
        }
    } else {
        false
    }
}

/// 类型推断
pub(crate) fn rust_type_to_json_type(ty: &Type) -> String {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            match segment.ident.to_string().as_str() {
                "String" | "str" => return "string".to_string(),
                "i32" | "i64" | "isize" | "u32" | "u64" | "usize" => return "integer".to_string(),
                "f32" | "f64" => return "number".to_string(),
                "bool" => return "boolean".to_string(),
                "Option" => {
                    if let syn::PathArguments::AngleBracketed(ref ab) = segment.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = ab.args.first() {
                            return rust_type_to_json_type(inner_ty);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    "string".to_string()
}
