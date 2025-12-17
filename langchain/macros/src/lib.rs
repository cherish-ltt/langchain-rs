use darling::{FromMeta, ast::NestedMeta};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, ItemFn, LitStr, parse_macro_input, punctuated::Punctuated};

mod tool;
use crate::tool::{ToolArgs, extract_params, rust_type_to_json_type};

#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let nested = match NestedMeta::parse_meta_list(args.into()) {
        Ok(n) => n,
        Err(e) => return e.to_compile_error().into(),
    };
    let tool_args = match ToolArgs::from_list(&nested) {
        Ok(a) => a,
        Err(e) => return e.write_errors().into(),
    };

    let input_fn = parse_macro_input!(input as ItemFn);

    if input_fn.sig.asyncness.is_none() {
        let err = syn::Error::new_spanned(&input_fn.sig.fn_token, "tool functions must be async");
        return err.to_compile_error().into();
    }

    let fn_name = input_fn.sig.ident.clone();
    let fn_vis = input_fn.vis.clone();
    let fn_inputs = input_fn.sig.inputs.clone().into_iter().collect::<Vec<_>>();

    let mut filtered_inputs = Vec::new();
    let mut has_state_param = false;

    for arg in &fn_inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            if !has_state_param {
                if let syn::Type::Reference(ref_ty) = &*pat_type.ty {
                    if let syn::Type::Path(type_path) = &*ref_ty.elem {
                        if let Some(segment) = type_path.path.segments.last() {
                            if segment.ident == "MessageState" {
                                has_state_param = true;
                                continue;
                            }
                        }
                    }
                }
            }
        }
        filtered_inputs.push(arg.clone());
    }

    let params = match extract_params(&filtered_inputs) {
        Ok(p) => p,
        Err(e) => return e.write_errors().into(),
    };

    let fn_name_str = fn_name.to_string();
    let mut base_struct_name = String::new();
    for part in fn_name_str.split('_') {
        if part.is_empty() {
            continue;
        }
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            base_struct_name.push(first.to_ascii_uppercase());
            base_struct_name.extend(chars);
        }
    }
    if base_struct_name.is_empty() {
        base_struct_name.push_str("Tool");
    }

    let tool_struct_ident = format_ident!("{}Tool", base_struct_name);
    let args_struct_ident = format_ident!("{}Args", base_struct_name);

    let fn_name_lit = LitStr::new(&fn_name_str, fn_name.span());
    let description_str = tool_args.description.unwrap_or_default();
    let description_lit = LitStr::new(&description_str, fn_name.span());

    let mut property_inserts = Vec::new();
    let mut required_fields = Vec::new();
    let mut arg_struct_fields = Vec::new();
    let mut call_args = Vec::new();

    for param in params {
        let param_name_str = param.name;
        let param_name_lit = LitStr::new(&param_name_str, fn_name.span());
        let param_ident = Ident::new(&param_name_str, fn_name.span());
        let param_ty = param.ty;
        let json_type = rust_type_to_json_type(&param_ty);
        let json_type_lit = LitStr::new(&json_type, fn_name.span());
        let arg_desc = tool_args
            .arg_descriptions
            .get(&param_name_str)
            .cloned()
            .unwrap_or_default();
        let arg_desc_lit = LitStr::new(&arg_desc, fn_name.span());

        arg_struct_fields.push(quote! {
            pub #param_ident: #param_ty
        });

        property_inserts.push(quote! {
            properties.insert(
                #param_name_lit.to_string(),
                serde_json::json!({
                    "type": #json_type_lit,
                    "description": #arg_desc_lit,
                }),
            );
        });

        if !param.is_optional {
            required_fields.push(quote! {
                #param_name_lit.to_string()
            });
        }

        call_args.push(quote! {
            parsed.#param_ident
        });
    }

    let state_arg = if has_state_param {
        quote! { _state, }
    } else {
        quote! {}
    };

    let expanded = quote! {
        #input_fn

        #[derive(serde::Deserialize)]
        #fn_vis struct #args_struct_ident {
            #(#arg_struct_fields,)*
        }

        #[derive(Debug, Clone)]
        #fn_vis struct #tool_struct_ident;

        #[async_trait::async_trait]
        impl crate::Tool for #tool_struct_ident {
            type Output = serde_json::Value;

            fn spec(&self) -> langchain_core::request::ToolSpec {
                let mut properties = serde_json::Map::new();
                #(#property_inserts)*
                let required: Vec<String> = vec![#(#required_fields),*];
                let parameters = serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                });

                langchain_core::request::ToolSpec::Function {
                    function: langchain_core::request::ToolFunction {
                        name: #fn_name_lit.to_owned(),
                        description: #description_lit.to_owned(),
                        parameters,
                    },
                }
            }

            async fn invoke(
                &self,
                _state: &langchain_core::state::MessageState,
                args: serde_json::Value,
            ) -> Result<Self::Output, langgraph::node::NodeRunError> {
                let parsed: #args_struct_ident = serde_json::from_value(args)
                    .map_err(|_| langgraph::node::NodeRunError::ToolRunError("args parse failed".to_string()))?;
                let result = #fn_name(
                    #state_arg
                    #(#call_args),*
                )
                .await
                .map_err(|_| langgraph::node::NodeRunError::ToolRunError("invoke failed".to_string()))?;
                serde_json::to_value(result)
                    .map_err(|_| langgraph::node::NodeRunError::ToolRunError("output serialize failed".to_string()))
            }
        }
    };

    expanded.into()
}

#[proc_macro]
pub fn tools_from_fns(input: TokenStream) -> TokenStream {
    let idents =
        parse_macro_input!(input with Punctuated::<Ident, syn::Token![,]>::parse_terminated);

    let mut tool_exprs = Vec::new();

    for fn_ident in idents {
        let fn_name_str = fn_ident.to_string();
        let mut base_struct_name = String::new();
        for part in fn_name_str.split('_') {
            if part.is_empty() {
                continue;
            }
            let mut chars = part.chars();
            if let Some(first) = chars.next() {
                base_struct_name.push(first.to_ascii_uppercase());
                base_struct_name.extend(chars);
            }
        }
        if base_struct_name.is_empty() {
            base_struct_name.push_str("Tool");
        }

        let tool_struct_ident = format_ident!("{}Tool", base_struct_name);

        tool_exprs.push(quote! {
            ::std::sync::Arc::new(#tool_struct_ident) as ::langchain::DynTool
        });
    }

    let expanded = quote! {
        vec![ #(#tool_exprs),* ]
    };

    expanded.into()
}
