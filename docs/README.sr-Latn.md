# Merchant Field Kit

[English](../README.md) · [Русский](README.ru.md) · **Srpski (latinica)**

Praktična analiza podataka za internet prodavnice. **Koliki prihod od prodaje možete očekivati u naredne četiri nedelje?** Prognozirajte prihod za planiranje prodaje i broj poručenih komada za odluke o zalihama. XYZ analiza pokazuje koji se proizvodi prodaju stabilno, a čiju prodaju je teže predvideti.

## Pokretanje na računaru

```sh
docker build -t merchant-field-kit .
docker run --rm -p 127.0.0.1:8501:8501 merchant-field-kit
```

Ako koristite Python 3.12, pogledajte [uputstvo za instalaciju i pokretanje](../README.md#run-locally). Otvorite [localhost:8501](http://localhost:8501), probajte primer prodavnice i odaberite prihod, komade ili oba pokazatelja. Prognozu za 28 dana možete preuzeti kao CSV. Internet je potreban za instalaciju, ne za izračunavanje. Kada alat pokrenete lokalno, datoteka ostaje na vašem računaru. Podaci o kupcima nisu potrebni.

## Šta možete da učitate

| Način unosa | Kolone | Primer |
|---|---|---|
| Dnevni zbir | `date`, `revenue` i/ili `units_ordered` | [Dnevni CSV](../examples/sample_shop.csv) |
| Stavke porudžbina | `date`, `sku`, `revenue` i/ili `units_ordered`. Ako imate broj komada, umesto `revenue` možete navesti `unit_price`. | [CSV sa stavkama](../examples/sample_orders.csv) |

Koristite CSV u UTF-8 formatu, veličine do 1 MB, sa najmanje **112 kalendarskih dana**. Dnevni niz ne sme imati praznine: upišite nulu samo za dane za koje znate da nije bilo prodaje. Kod stavki porudžbina dani bez redova između prvog i poslednjeg datuma računaju se kao nula, zato izvoz mora biti potpun. Nemojte mešati valute, prihod i dobit ili iznose pre i posle popusta. Obradite povraćaje pre uvoza. Koristite lokalne datume prodavnice. ISO je najsigurniji format; ako datum može da se protumači na dva načina, izaberite redosled dana i meseca.

## Kako čitati rezultat

Prophet se poredi sa ponavljanjem poslednjih sedam dana i prosekom poslednjih 28 dana. Pri testiranju na istorijskim periodima svaki model koristi samo podatke dostupne pre početka perioda. Greška je izražena u valuti po danu ili u komadima po danu. Prognoza prihoda nije prognoza dobiti, a procena komada nije predlog količine za nabavku. Akcije, nedostatak robe i promene asortimana mogu da utiču na obe prognoze.

Za XYZ analizu potrebne su stavke porudžbina sa šifrom proizvoda i brojem komada. Analiza poredi nedeljni broj poručenih komada po šifri proizvoda (SKU), za pune nedelje od ponedeljka do nedelje. X označava stabilniju prodaju, Y promenljiviju, a Z najpromenljiviju. To nije rangiranje po prihodu. Proizvod bez dovoljno podataka ostaje bez klase.

[Poređenje na podacima stvarne prodavnice](benchmark.md) odnosi se samo na poručene komade. Ne potvrđuje tačnost prognoze prihoda niti rast prodaje.
