use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{DeriveInput, parse_macro_input, spanned::Spanned};

#[proc_macro_derive(GraphLabel)]
pub fn graph_label_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    derive_graph_label_impl(input)
}

fn derive_graph_label_impl(input: DeriveInput) -> TokenStream {
    if let syn::Data::Union(_) = &input.data {
        let message = "Cannot derive GraphLabel for unions.";
        return quote_spanned! {
            input.span() => compile_error!(#message);
        }
        .into();
    }

    let ident = &input.ident;

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let mut where_clause = where_clause.cloned().unwrap_or_else(|| syn::WhereClause {
        where_token: Default::default(),
        predicates: Default::default(),
    });
    where_clause.predicates.push(
        syn::parse2(quote! {
            Self: 'static + Send + Sync + Clone + Eq + ::core::fmt::Debug + ::core::hash::Hash
        })
        .unwrap(),
    );

    // --- as_str() 实现 ---
    let as_str_impl = match &input.data {
        syn::Data::Enum(data_enum) => {
            let arms: Vec<_> = data_enum
                .variants
                .iter()
                .map(|variant| {
                    let v_ident = &variant.ident;
                    let s = v_ident.to_string();
                    quote! {
                        #ident::#v_ident { .. } => #s,
                    }
                })
                .collect();

            quote! {
                fn as_str(&self) -> &'static str {
                    match self {
                        #(#arms)*
                    }
                }
            }
        }
        syn::Data::Struct(_) => {
            let struct_name = ident.to_string();
            quote! {
                fn as_str(&self) -> &'static str {
                    #struct_name
                }
            }
        }
        _ => quote! {}, // union 已经提前返回
    };

    quote! {
        // To ensure alloc is available, but also prevent its name from clashing, we place the implementation inside an anonymous constant
        // 把 extern crate alloc 和 impl 块包裹在一个局部作用域中，避免在用户模块顶层直接引入名为 alloc 的 crate，和用户自己的 use alloc 等名字起冲突。
        const _: () = {
            extern crate alloc;

            impl #impl_generics GraphLabel for #ident #ty_generics #where_clause {
                fn dyn_clone(&self) -> alloc::boxed::Box<dyn GraphLabel> {
                    alloc::boxed::Box::new(::core::clone::Clone::clone(self))
                }

                #as_str_impl
            }
        };
    }
    .into()
}
